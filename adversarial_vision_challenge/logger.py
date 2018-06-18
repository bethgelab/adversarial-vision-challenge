import logging
import os

FORMAT = '%(asctime)-15s %(message)s'
filename = os.getenv('LOG_FILE', '/tmp/avc_log.txt')
logging.basicConfig(filename=filename, format=FORMAT)
logger = logging.getLogger('adversarial_vision_challenge')
logger.setLevel(logging.DEBUG)