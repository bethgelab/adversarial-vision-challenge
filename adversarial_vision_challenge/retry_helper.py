import time
from .logger import logger

class RetriesExceededError(Exception):
    pass


def retryable(func, retries=3):
    def retry(*args, **kwargs):
        retried = 0
        backoff = 1

        while retried < 3:
            try:
                return func(*args, **kwargs)
            except:
                if retried < retries:
                    time.sleep(3 * backoff)
                    retried += 1
                    backoff += 1
                    logger.info('Retrying for the %sth time', retried)
                else:
                    logger.error('Retried request for %s times. Giving up.', retried)
                    raise RetriesExceededError(
                        "Failed already {0} times. No further retrying.".format(retries))

    return retry
