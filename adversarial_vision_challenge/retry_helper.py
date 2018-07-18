import requests
from functools import wraps
import time
from .logger import logger
from .notifier import CrowdAiNotifier


class RetriesExceededError(Exception):
    pass


def retryable(func, retries=3):
    @wraps(func)
    def retry(*args, **kwargs):
        for retried in range(retries + 1):
            if retried > 0:
                logger.info('Retrying for the %s. time', retried)
                time.sleep(3 * retried)
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException:
                pass
        logger.error('Retried request for %s times. Giving up.', retried)
        CrowdAiNotifier.retries_exceeded()
        raise RetriesExceededError(
            "Failed already {0} times. No further retrying.".format(retries))

    return retry
