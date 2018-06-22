import logging
import os

filename = os.getenv('LOG_FILE')
logger = None

if filename is not None:
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=filename, format=FORMAT)
    logger = logging.getLogger('adversarial_vision_challenge')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

