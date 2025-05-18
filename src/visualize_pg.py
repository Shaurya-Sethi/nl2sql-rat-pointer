import torch
import matplotlib.pyplot as plt
import numpy as np
from decoder_pg import PointerGeneratorDecoder
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from config import NL2SQLConfig
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_pointer_generator(config_path, tokenizer_path=None, schema_sample=None, use_random=False):
    """
    Visualize the pointer-generator mechanism's copy vs. generate probabilities.
    
    Args:
        config_path: Path to configuration
        tokenizer_path: Path to tokenizer (if different from config)
        schema_sample: Sample schema text to use
        use_random: Whether to use random tokens for visualization
    """
    # Load config
    config = NL2SQLConfig.from_yaml(config_path, 'sft')
    
    # Set up tokenizer
    if tokenizer_path:
        tokenizer = NL2SQLTokenizer(tokenizer_path, config.special_tokens)
    else:
        tokenizer = NL2SQLTokenizer(config.sp_model_path, config.special_tokens)
    
    # Create small model for visualization
    model = NL2SQLTransformer(
        vocab_size=config.vocab_size,
        d_model=128,  # Small model for visualization
        n_heads=4,
        n_layers=2,
        num_relations=config.num_relations,
        dropout=0.0,
        use_pointer_generator=True,
        pad_token_id=config.pad_token_id
    )
    
    # Create or use sample schema
    if not schema_sample and not use_random:
        schema_sample = f"{config.special_tokens['SCHEMA_START']} Users(id: int PRIMARY KEY, {config.special_tokens['PK_START']}name{config.special_tokens['PK_END']}: string, email: string) Posts(id: int PRIMARY KEY, {config.special_tokens['FK_START']}user_id -> Users.id{config.special_tokens['FK_END']}: int, title: string, content: text) {config.special_tokens['SCHEMA_END']}"
    
    if use_random:
        # Create random data
        batch_size = 1
        src_len = 20
        tgt_len = 10
        
        src_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
        tgt_ids = torch.randint(0, config.vocab_size, (batch_size, tgt_len))
        schema_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
        schema_mask[0, 5:10] = True  # Mark positions 5-9 as schema tokens
        relation_ids = torch.zeros(batch_size, src_len, src_len, dtype=torch.long)
    else:
        # Encode the sample schema
        schema_tokens = tokenizer.encode(schema_sample)
        question = f"{config.special_tokens['NL_START']} Find all posts by user named 'John' {config.special_tokens['NL_END']}"
        question_tokens = tokenizer.encode(question)
        
        # Create input sequence
        src_tokens = schema_tokens + question_tokens
        src_ids = torch.tensor([src_tokens], dtype=torch.long)
        
        # Create a sample partial SQL query
        sql_prefix = f"{config.special_tokens['SQL_START']} SELECT Posts.title FROM Posts JOIN "
        tgt_tokens = tokenizer.encode(sql_prefix)
        tgt_ids = torch.tensor([tgt_tokens], dtype=torch.long)
        
        # Create schema mask - mark schema tokens
        schema_mask = torch.zeros(1, len(src_tokens), dtype=torch.bool)
        schema_end_idx = schema_tokens.index(tokenizer.get_special_token_id('SCHEMA_END'))
        schema_mask[0, 1:schema_end_idx] = True  # Mark everything in schema section (except start/end tags)
        
        # Create dummy relation matrix
        relation_ids = torch.zeros(1, len(src_tokens), len(src_tokens), dtype=torch.long)
    
    # Run model forward pass to get decoder outputs
    with torch.no_grad():
        encoder_output = model.encoder(
            input_ids=src_ids,
            relation_ids=relation_ids,
            attention_mask=None
        )
        
        # Access the p_gen values directly from decoder 
        decoder = model.decoder
        B, T = tgt_ids.shape
        
        # Get embeddings
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = decoder.token_emb(tgt_ids) + decoder.pos_emb(positions)
        
        # Track attention weights and p_gen values
        attention_weights = []
        p_gen_values = []
        
        # Pass through decoder layers
        attn_w_last = None
        for layer in decoder.layers:
            x, attn_w = layer(
                x,
                encoder_output,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
            )
            attn_w_last = attn_w
            attention_weights.append(attn_w.cpu().numpy())
        
        # Calculate p_gen at each step
        schema_mask_f = schema_mask.to(dtype=x.dtype)
        P_copy_src = attn_w_last * schema_mask_f.unsqueeze(1)
        denom = P_copy_src.sum(-1, keepdim=True) + 1e-8
        P_copy_src = P_copy_src / denom
        
        # Compute context vector
        c_t = torch.bmm(P_copy_src, encoder_output)
        
        # Calculate p_gen values
        gate_inp = torch.cat([x, c_t, decoder.token_emb(tgt_ids)], dim=-1)
        p_gen = torch.sigmoid(decoder.p_gen(gate_inp))
        p_gen_values = p_gen.squeeze(-1).cpu().numpy()
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot attention heatmap
    plt.subplot(2, 1, 1)
    attn_heatmap = attention_weights[-1][0]  # Take last layer's attention weights
    plt.imshow(attn_heatmap, cmap='viridis')
    plt.colorbar(label='Attention weight')
    plt.title('Cross-Attention Weights')
    plt.xlabel('Source position')
    plt.ylabel('Target position')
    
    # Mark schema positions
    schema_positions = np.where(schema_mask[0].cpu().numpy())[0]
    if len(schema_positions) > 0:
        plt.axvspan(schema_positions.min() - 0.5, schema_positions.max() + 0.5, 
                  alpha=0.2, color='red', label='Schema tokens')
    
    # Plot p_gen values
    plt.subplot(2, 1, 2)
    plt.plot(p_gen_values[0], 'bo-', label='p_gen (generate probability)')
    plt.plot(1 - p_gen_values[0], 'ro-', label='1-p_gen (copy probability)')
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.title('Generate vs. Copy Probabilities')
    plt.xlabel('Target position')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'pointer_generator_visualization.png')
    logger.info(f"Visualization saved to {output_dir / 'pointer_generator_visualization.png'}")
    
    # Display tokens and their generate/copy probabilities
    if not use_random:
        logger.info("\nGenerate/Copy probabilities per token:")
        for i in range(len(tgt_tokens)):
            token = tokenizer.decode([tgt_tokens[i]])
            logger.info(f"Position {i}: '{token}' - Generate: {p_gen_values[0][i]:.4f}, Copy: {1-p_gen_values[0][i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize pointer-generator mechanism')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--tokenizer', type=str, default=None,
                      help='Path to tokenizer (optional)')
    parser.add_argument('--random', action='store_true',
                      help='Use random data for visualization')
    args = parser.parse_args()
    
    visualize_pointer_generator(args.config, args.tokenizer, use_random=args.random) 