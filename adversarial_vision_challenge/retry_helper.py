import time


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
                else:
                    raise RetriesExceededError(
                        "Failed already {0} times. No further retrying.".format(retries))

    return retry
