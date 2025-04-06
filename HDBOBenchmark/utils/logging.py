import logging
from logging import Logger as _Logger
import time


class Logger(_Logger):
    def __init__(self, name, level=0):
        super().__init__(name, level)
        self.start_time = time.time()

    def _process(self, msg):
        return f"[{time_text(time.time() - self.start_time)}] " + msg

    def callHandlers(self, record):
        if not "msg_type" in record.__dict__:
            record.msg = self._process(record.msg)
        super().callHandlers(record)


logging.basicConfig(level="INFO", format="%(message)s")
logging.setLoggerClass(Logger)
logger = logging.getLogger(__name__)


def time_dhms(s):
    s = int(s)
    day = s // 86400
    s %= 86400
    hour = s // 3600
    s %= 3600
    minute = s // 60
    s %= 60
    return day, hour, minute, s


def time_text(s):
    d, h, m, s = time_dhms(s)
    return f"{d}d {h:>2d}:{m:>2d}:{s:>2d}"
