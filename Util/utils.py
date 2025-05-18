import os
import logging
import datetime
from typing import Optional

def setup_logger(
    name: str = "qwen_distill",
    log_level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger for the project.
    
    Args:
        name: Logger name
        log_level: Logging level (default: INFO)
        log_dir: Directory to save log files (default: ./logs)
        console_output: Whether to output logs to console
        file_output: Whether to save logs to file
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Format with timestamp, level, and message
    log_format = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    logger.info(f"Logger '{name}' initialized")
    return logger 