import logging
import sys

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Set up a logger that writes to both console and a file (if provided)
    
    Args:
        log_file (str, optional): Path to log file. If None, only console logging is used.
        log_level (int, optional): Logging level. Default is logging.INFO.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('battery_prediction')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers if any
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}")
            # Continue without file logging
    
    return logger 