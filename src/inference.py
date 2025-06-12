"""
Inference module for NL2SQL Transformer.
----------------------------------------
- Loads model and tokenizer from provided paths.
- Accepts a user question and a pre-tokenized schema (as a list of token IDs).
- Wraps the question in <NL> ... </NL> tokens, tokenizes, and concatenates with schema tokens.
- Runs the model in eval mode, generates output tokens, and decodes to text.
- Extracts Chain-of-Thought (CoT) and SQL segments from the output.
- Returns output as a dict: {"cot": "...", "sql": "..."} or {"sql": "..."}.
"""

import os
import torch
from typing import List, Dict, Optional, Union, Iterator

from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from config import NL2SQLConfig
from relation_matrix import RelationMatrixBuilder
from schema_pipeline.formatter import SPECIAL_TOKENS

class NL2SQLInference:
    """
    Production-ready inference class for NL2SQL Transformer.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        max_gen_tokens: int = 1024,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model checkpoint (.pt or .bin)
            config_path: Path to YAML config file (for model/tokenizer params)
            tokenizer_path: Optional path to SentencePiece model (overrides config)
            device: 'cuda', 'cpu', or None (auto)
            max_gen_tokens: Maximum tokens to generate
        """
        # Load config
        self.config = NL2SQLConfig.from_yaml(config_path, phase='sft')
        self.max_gen_tokens = max_gen_tokens

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        sp_model_path = tokenizer_path or self.config.sp_model_path
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"Tokenizer model not found at: {sp_model_path}")
        self.tokenizer = NL2SQLTokenizer(sp_model_path, self.config.special_tokens)

        # Load model
        # Pass COT_START and SQL_END token IDs for correct generation
        cot_start_token_id = self.tokenizer.get_special_token_id("COT_START")
        sql_end_token_id = self.tokenizer.get_special_token_id("SQL_END")
        self.model = NL2SQLTransformer(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            num_relations=self.config.num_relations,
            dropout=self.config.dropout,
            max_len=self.config.max_len,
            use_pointer_generator=self.config.use_pointer_generator,
            pad_token_id=self.config.pad_token_id,
            cot_start_token_id=cot_start_token_id, # Use <COT> as start of generation
            sql_end_token_id=sql_end_token_id     # Use </SQL> as end of generation
        )
        self.model.to(self.device)
        self.model.eval()

        # Load model weights
        self._load_weights(model_path)

        # For pointer-generator schema mask
        if self.config.use_pointer_generator:
            self.relation_builder = RelationMatrixBuilder(
                tokenizer=self.tokenizer,
                num_relations=self.config.num_relations
            )
        else:
            self.relation_builder = None

    def _load_weights(self, model_path: str):
        """
        Load model weights from checkpoint.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        # Support both Trainer checkpoints and plain state_dicts
        if isinstance(state, dict) and 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)

    def _prepare_model_inputs(self, question: str, schema_tokens: Union[List[int], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Helper method to prepare model inputs for both infer and infer_stream.
        """
        # 1. Tokenize NL question with <NL> ... </NL> wrappers
        nl_start = self.tokenizer.get_special_token_id("NL_START")
        nl_end = self.tokenizer.get_special_token_id("NL_END")
        nl_ids = [nl_start] + self.tokenizer.encode(question, add_special_tokens=False) + [nl_end]

        # 2. Concatenate NL and schema tokens
        if isinstance(schema_tokens, torch.Tensor):
            schema_ids = schema_tokens.tolist()
        else:
            schema_ids = list(schema_tokens)
        encoder_input_ids_list = nl_ids + schema_ids
        input_len = len(encoder_input_ids_list)

        if input_len > self.config.max_len:
            raise ValueError(
                f"Combined token length of NL ({len(nl_ids)}) and schema ({len(schema_ids)}) tokens is {input_len}, "
                f"which exceeds the model's configured max_len ({self.config.max_len}). "
                f"Please shorten the question or provide a smaller schema segment."
            )

        # 3. Prepare model inputs (tensors)
        encoder_input_ids = torch.tensor(encoder_input_ids_list, dtype=torch.long).unsqueeze(0).to(self.device)
        encoder_attention_mask = (encoder_input_ids != self.config.pad_token_id).to(self.device)  # True for non-pad

        relation_matrix = self.relation_builder.build_relation_matrix(encoder_input_ids_list)
        relation_matrix = relation_matrix.to(self.device)
        relation_matrix = relation_matrix.unsqueeze(0)  # (1, L, L)

        # Build schema_mask for pointer-generator
        schema_mask = None
        if self.config.use_pointer_generator:
            schema_mask = torch.zeros((1, input_len), dtype=torch.bool, device=self.device)
            try:
                schema_start_id = self.tokenizer.get_special_token_id("SCHEMA_START")
                schema_end_id = self.tokenizer.get_special_token_id("SCHEMA_END")
                schema_start_idx = encoder_input_ids_list.index(schema_start_id)
                schema_end_idx = encoder_input_ids_list.index(schema_end_id)
                if schema_end_idx > schema_start_idx:
                    schema_mask[0, schema_start_idx:schema_end_idx+1] = True
            except ValueError:
                # If schema tags not found, leave mask as all False
                pass

        # Return all model inputs, all on self.device
        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "relation_matrix": relation_matrix,
            "schema_mask": schema_mask,
        }

    def infer(
        self,
        question: str,
        schema_tokens: Union[List[int], torch.Tensor],
    ) -> Dict[str, str]:
        """
        Run inference for a single NL question and pre-tokenized schema.

        Args:
            question: User's natural language question (str)
            schema_tokens: List[int] or torch.Tensor of token IDs for the schema (<SCHEMA> ... </SCHEMA>)

        Returns:
            Dict with keys "sql" and optionally "cot".
        """
        # 1. Prepare model inputs using helper
        inputs = self._prepare_model_inputs(question, schema_tokens)
        encoder_input = inputs["encoder_input_ids"]
        relation_matrix = inputs["relation_matrix"]
        encoder_attention_mask = inputs["encoder_attention_mask"]
        schema_mask = inputs["schema_mask"]

        # 4. Generate output tokens
        with torch.no_grad():
            gen_ids = self.model.generate(
                encoder_input_ids=encoder_input,
                encoder_relation_ids=relation_matrix,
                max_length=self.max_gen_tokens,
                encoder_attention_mask=encoder_attention_mask,
                schema_mask=schema_mask,
            )
        # Remove batch dimension
        gen_ids = gen_ids[0].tolist()

        # 5. Parse output: extract <COT>...</COT> and <SQL>...</SQL>
        return self._parse_output(gen_ids)

    def _parse_output(self, output_ids: List[int]) -> Dict[str, str]:
        """
        Extract CoT and SQL segments from generated token IDs.
        """
        # TEMP DEBUG STATEMENTS
        # print("DEBUG: output_ids =", output_ids)
        # print("DEBUG: SQL_START token ID:", self.tokenizer.get_special_token_id("SQL_START"))
        # print("DEBUG: SQL_END token ID:", self.tokenizer.get_special_token_id("SQL_END"))
        # print("DEBUG: Decoded full output:", self.tokenizer.decode(output_ids, skip_special_tokens=False))
        # print("DEBUG: COT_START token ID:", self.tokenizer.get_special_token_id("COT_START"))
        # print("DEBUG: COT_END token ID:", self.tokenizer.get_special_token_id("COT_END"))


        # Map token IDs to text
        id2tok = {v: k for k, v in self.tokenizer.special_token_ids.items()}
        # Find segment indices
        def find_segment(start_name, end_name):
            try:
                start = output_ids.index(self.tokenizer.get_special_token_id(start_name))
                end = output_ids.index(self.tokenizer.get_special_token_id(end_name), start + 1)
                return start, end
            except ValueError:
                return None, None

        cot_start, cot_end = find_segment("COT_START", "COT_END")
        sql_start, sql_end = find_segment("SQL_START", "SQL_END")

        # Extract segments
        result = {}
        if cot_start is not None and cot_end is not None:
            cot_ids = output_ids[cot_start + 1: cot_end]
            result["cot"] = self.tokenizer.decode(cot_ids, skip_special_tokens=True).strip()
        if sql_start is not None and sql_end is not None:
            sql_ids = output_ids[sql_start + 1: sql_end]
            result["sql"] = self.tokenizer.decode(sql_ids, skip_special_tokens=True).strip()
        elif sql_start is not None:
            # If only <SQL_START> found, decode until end or next special
            sql_ids = output_ids[sql_start + 1:]
            result["sql"] = self.tokenizer.decode(sql_ids, skip_special_tokens=True).strip()
        else:
            # Fallback: try to decode everything after <NL_END> or just the whole output
            try:
                nl_end = output_ids.index(self.tokenizer.get_special_token_id("NL_END"))
                after_nl = output_ids[nl_end + 1:]
                result["sql"] = self.tokenizer.decode(after_nl, skip_special_tokens=True).strip()
            except Exception:
                result["sql"] = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return result

    def infer_stream(
        self,
        question: str,
        schema_tokens: Union[List[int], torch.Tensor],
    ) -> Iterator[str]:
        """
        Token-by-token streaming inference.
        Yields each decoded token string as soon as it is produced.
        """
        # 1. Build the encoder inputs exactly as in `infer()` using helper
        inputs = self._prepare_model_inputs(question, schema_tokens)
        encoder_input = inputs["encoder_input_ids"]
        relation_matrix = inputs["relation_matrix"]
        encoder_attention_mask = inputs["encoder_attention_mask"]
        schema_mask = inputs["schema_mask"]

        # 2. Call the streaming generate:
        gen_iter = self.model.generate(
            encoder_input_ids=encoder_input,
            encoder_relation_ids=relation_matrix,
            max_length=self.max_gen_tokens,
            encoder_attention_mask=encoder_attention_mask,
            schema_mask=schema_mask,
            stream=True
        )

        # 3. Decode & yield
        for token_id in gen_iter:
            txt = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            yield txt

# ------------------- Usage Example -------------------

if __name__ == "__main__":
    # Example usage (replace paths with actual ones)
    import argparse
    import json

    parser = argparse.ArgumentParser(description="NL2SQL Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to SentencePiece model (optional)")
    parser.add_argument("--schema_tokens", type=str, required=True, help="Path to file with schema token IDs (JSON list)")
    parser.add_argument("--question", type=str, required=True, help="Natural language question")
    args = parser.parse_args()

    # Load schema tokens (expects a JSON list of ints)
    with open(args.schema_tokens, "r") as f:
        schema_tokens = json.load(f)

    engine = NL2SQLInference(
        model_path=args.model,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
    )
    result = engine.infer(args.question, schema_tokens)
    print(json.dumps(result, indent=2))