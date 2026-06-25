import logging


# Set root logger to WARNING — suppresses noisy third-party library logs (OpenAI, httpcore etc.)
logging.basicConfig(
   level=logging.WARNING,
   format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
   # Our own loggers run at DEBUG — only third-party libraries are suppressed
   logger = logging.getLogger(name)
   logger.setLevel(logging.DEBUG)
   return logger
