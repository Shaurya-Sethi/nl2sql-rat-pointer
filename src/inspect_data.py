import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_sft_data(data_file: str, num_samples: int = 2):
    """
    Inspect samples from the SFT dataset.
    
    Args:
        data_file: Path to the SFT data file
        num_samples: Number of samples to inspect
    """
    try:
        # Read and parse samples
        with open(data_file, 'r', encoding='utf-8') as f:
            # Read first few lines for debugging
            lines = []
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                lines.append(line.strip())
                
        logger.info(f"First {len(lines)} lines from file:")
        for i, line in enumerate(lines):
            logger.info(f"\nLine {i+1}:")
            logger.info(line[:200] + "..." if len(line) > 200 else line)
            
        # Try to parse as JSON
        try:
            samples = [json.loads(line) for line in lines]
            logger.info("\nSuccessfully parsed as JSON")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse as JSON: {e}")
            # Try to parse as plain text
            logger.info("\nTrying to parse as plain text...")
            for i, line in enumerate(lines):
                logger.info(f"\nSample {i+1}:")
                logger.info(line)
            
    except Exception as e:
        logger.error(f"Error inspecting data: {e}")
        raise

def inspect_txt_data(data_file: str, num_samples: int = 2):
    """
    Inspect samples from a plain text dataset.

    Args:
        data_file: Path to the data file
        num_samples: Number of samples to inspect
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                lines.append(line.rstrip('\n'))

        print(f"First {len(lines)} lines from file:")
        for i, line in enumerate(lines):
            print(f"\nLine {i+1} (length {len(line)}):")
            print(line[:500] + ("..." if len(line) > 500 else ""))

    except Exception as e:
        print(f"Error inspecting data: {e}")

def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Path to SFT data file
    sft_file = project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_train.txt"
    
    if not sft_file.exists():
        logger.error(f"Data file not found: {sft_file}")
        return
        
    # Inspect data
    inspect_sft_data(str(sft_file))

    # Inspect plain text data
    inspect_txt_data(str(sft_file))

if __name__ == '__main__':
    main() 