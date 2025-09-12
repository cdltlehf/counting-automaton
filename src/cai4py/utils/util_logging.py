import logging

debugger_disabled = True # or dynamically set this from env/config
def setup_debugger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # Prevent duplicate handlers on reload
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.disabled = debugger_disabled
    return logger