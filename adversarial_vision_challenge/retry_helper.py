import time


class RetriesExceededError(Exception):
    pass

def retryable(func, retries=3):
   def retry(retried=0, backoff=1):
        try:
           return func()
        except:
            if retried < retries:
                time.sleep(3 * backoff)
                retry(retried + 1, backoff + 1)
            else:
                raise RetriesExceededError("Failed already {0} times. No further retrying.".format(retries))
   return retry