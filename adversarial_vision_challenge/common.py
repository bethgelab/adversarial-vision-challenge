import numpy as np
from .logger import logger


def check_image(image):
    # image should a 64 x 64 x 3 RGB image
    assert isinstance(
        image, np.ndarray), "image should be an numpy array"
    assert image.shape == (64, 64, 3), "image should be of size 64x64x3"
    if image.dtype == np.float32:
        # we accept float32, but only if the values
        # are between 0 and 255 and we convert them
        # to integers
        if image.min() < 0:
            logger.warning('clipped value smaller than 0 to 0')
        if image.max() > 255:
            logger.warning('clipped value greater than 255 to 255')
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    assert image.dtype == np.uint8
    return image
