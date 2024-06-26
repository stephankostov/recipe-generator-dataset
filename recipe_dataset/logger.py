# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/99-logger.ipynb.

# %% auto 0
__all__ = ['configure_logger']

# %% ../notebooks/99-logger.ipynb 1
import logging

def configure_logger(logger):

    default_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s():%(lineno)s] [PID:%(process)d TID:%(thread)d] %(message)s", 
    "%d/%m/%Y %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(default_formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    
    return logger
