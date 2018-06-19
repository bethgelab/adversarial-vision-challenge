import requests
from functools import wraps
import time
from .logger import logger
import traceback
import sys


class RetriesExceededError(Exception):
    pass


def retryable(func, retries=3):
    @wraps(func)
    def retry(*args, **kwargs):
        for retried in range(retries):
            if retried > 0:
                logger.info('Retrying for the %sth time', retried)
                time.sleep(3 * retried)
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException:
                traceback.print_exc(file=sys.stdout)
        logger.error('Retried request for %s times. Giving up.', retried)
        raise RetriesExceededError(
            "Failed already {0} times. No further retrying.".format(retries))
    return retry
