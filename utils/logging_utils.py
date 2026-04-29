import logging

class OneTimeFilter(logging.Filter):
    """
    A logging filter that only allows each unique message to be logged once.
    Prevents console flooding in loops (e.g., model layers).
    """
    def __init__(self):
        super().__init__()
        self.logged_messages = set()

    def filter(self, record):
        message = record.getMessage()
        if message in self.logged_messages:
            return False
        self.logged_messages.add(message)
        return True

def get_one_time_logger(name):
    """
    Returns a logger with a OneTimeFilter attached.
    """
    logger = logging.getLogger(name)
    # Avoid adding multiple filters if called multiple times
    if not any(isinstance(f, OneTimeFilter) for f in logger.filters):
        logger.addFilter(OneTimeFilter())
    return logger
