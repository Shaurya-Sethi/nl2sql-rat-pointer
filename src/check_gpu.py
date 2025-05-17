import torch
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"Current GPU device: {torch.cuda.current_device()}")
        logger.info(f"GPU device count: {torch.cuda.device_count()}")
    else:
        logger.error("CUDA is not available. Possible reasons:")
        logger.error("1. PyTorch was not installed with CUDA support")
        logger.error("2. CUDA drivers are not properly installed")
        logger.error("3. GPU is not compatible with current CUDA version")
        
        # Check if CUDA is in PATH
        import os
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            logger.info(f"CUDA_PATH is set to: {cuda_path}")
        else:
            logger.error("CUDA_PATH is not set in environment variables")

if __name__ == '__main__':
    check_gpu() 