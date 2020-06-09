import logging
import sys


def create_logger(logging_path):
    # Create a custom logger
    logger = logging.getLogger("logger and handler")
    # set logger level
    logger.setLevel(logging.INFO)
    # Create handlers
    # c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(logging_path)
    # set level for each handler
    # c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    # c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    # logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger




