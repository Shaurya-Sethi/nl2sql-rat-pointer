import torch
import sys
import site
import logging
import os

# Set up logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for better readability
)
logger = logging.getLogger(__name__)

def check_torch_installation():
    logger.info("=== PyTorch Installation Details ===")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"PyTorch path: {torch.__file__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check all site-packages directories
    logger.info("\n=== Site-packages Directories ===")
    for path in site.getsitepackages():
        logger.info(f"  {path}")
    
    # Check for torch installations
    logger.info("\n=== PyTorch Installations Found ===")
    for path in site.getsitepackages():
        torch_path = os.path.join(path, 'torch')
        if os.path.exists(torch_path):
            logger.info(f"Found torch in: {torch_path}")
            # Try to get version from this installation
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("torch", os.path.join(torch_path, "__init__.py"))
                torch_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(torch_module)
                logger.info(f"  Version: {torch_module.__version__}")
            except Exception as e:
                logger.error(f"  Error checking version: {e}")

if __name__ == '__main__':
    check_torch_installation() 