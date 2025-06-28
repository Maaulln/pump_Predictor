"""
Enhanced logging utilities
"""
import sys
from pathlib import Path
from loguru import logger
from pump_predictor.config import LOG_DIR, LOGGING_CONFIG

def setup_logger():
    """Setup loguru logger with file and console output"""
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format=LOGGING_CONFIG['format'],
        level=LOGGING_CONFIG['level'],
        colorize=True
    )
    
    # File handler
    log_file = LOG_DIR / "pump_predictor.log"
    logger.add(
        log_file,
        format=LOGGING_CONFIG['format'],
        level=LOGGING_CONFIG['level'],
        rotation=LOGGING_CONFIG['rotation'],
        retention=LOGGING_CONFIG['retention'],
        compression="zip"
    )
    
    return logger

def get_logger(name: str = None):
    """Get logger instance"""
    if name:
        return logger.bind(name=name)
    return logger

# Initialize logger on import
setup_logger()