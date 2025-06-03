"""
Logging configuration
"""
import logging
from pathlib import Path
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """Configure and return logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"pump_predictor_{datetime.now():%Y%m%d}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger