import logging


class LogFormatter(logging.Formatter):
    # color reference:
    # https://talyian.github.io/ansicolors/
    # https://bixense.com/clicolors/
    def __init__(self):
        super(LogFormatter, self).__init__()

        self.GREEN   = '\x1b[38;5;119m'
        self.CYAN    = '\x1b[38;5;87m'
        self.YELLOW  = '\x1b[38;5;226m'
        self.RED     = '\x1b[38;5;196m'
        self.RESET   = '\x1b[0m'
        self.PREFIX  = 'generate'
        self.TIME    = '%(asctime)s.%(msecs)03d'
        self.LEVEL   = '%(levelname)s'
        self.MESSAGE = '%(message)s'

        self.mapping = {
            logging.DEBUG    : self.GREEN,
            logging.INFO     : self.CYAN,
            logging.WARNING  : self.YELLOW,
            logging.ERROR    : self.RED,
            logging.CRITICAL : self.RED,
        }

    def colored_fmt(self, color):
        return f'{color}[{self.TIME}] [{self.LEVEL}] {self.MESSAGE}{self.RESET}'

    def format(self, record):
        log_fmt = self.colored_fmt(self.mapping.get(record.levelno))
        formatter = logging.Formatter(log_fmt, datefmt='%y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(logging_level, debug):
    if logging_level is None:
        if debug:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

    elif logging_level == 'debug':
        logging_level = logging.DEBUG

    elif logging_level == 'info':
        logging_level = logging.INFO

    elif logging_level == 'warning':
        logging_level = logging.WARNING

    elif logging_level == 'error':
        logging_level = logging.ERROR

    else:
        # we cannot use us.raise_exception here because it relies on the logger
        raise Exception('Unsupported logging_level.')

    logger = logging.getLogger('unisim')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging_level)
        ch.setFormatter(LogFormatter())
        logger.addHandler(ch)
        logger.propagate = False  # prevent double logging

    return logger
