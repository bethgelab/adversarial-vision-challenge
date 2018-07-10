import numpy as np
from .logger import logger
import json
import os
import uuid
from .notifier import CrowdAiNotifier


def check_image(image):
    # image should a 64 x 64 x 3 RGB image
    _assert(isinstance(
        image, np.ndarray), "image should be an numpy array")
    _assert(image.shape == (64, 64, 3), "image should be of size 64x64x3")
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

def check_track(directory, track):
    crowdai_json = os.path.join(directory, "crowdai.json")
    with open(crowdai_json) as file:
        data = json.load(file)
    id = data['challenge_id']

    _assert(track == id, "you are running test script for {0}, but the crowdai.json says: {1}".format(track, id))


def _assert(condition, message):
    try:
        assert condition, message
    except AssertionError as e:
        CrowdAiNotifier.assertion_failure(message)
        raise e

def reset_repo2docker_cache():
    """
        repo2docker does not support ignoring the repo2docker cache yet,
        so we work around that by adding some content to the repository to
        reset the cache.
    """
    crowdai_folder = ".crowdai"
    if not os.path.exists(crowdai_folder):
        os.mkdir(crowdai_folder)

    fp = open(os.path.join(
        crowdai_folder, "state"
    ), "w")
    fp.write(str(uuid.uuid4()))
    fp.close()
