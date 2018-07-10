import os

import numpy as np
import yaml

from .client import TinyImageNetBSONModel
from .logger import logger
from .notifier import CrowdAiNotifier
from .common import check_image

from adversarial_vision_challenge.retry_helper import RetriesExceededError


def _load_img(path):
    """
        Reads image image from the given path and returns an numpy array.
    """
    path = os.path.join(os.path.dirname(__file__), path)
    image = np.load(path)
    assert image.dtype == np.uint8
    assert image.shape == (64, 64, 3)
    return image


def _read_image(file_name):
    """
        Returns a tuple of image as numpy array and label as int,
        given the csv row.
    """
    input_folder = os.getenv('INPUT_IMG_PATH')
    img_path = os.path.join(input_folder, file_name)
    image = _load_img(img_path)
    assert image.dtype == np.uint8
    image = image.astype(np.float32)
    assert image.dtype == np.float32
    return image


def read_images():
    """
        Returns a list containing tuples of images as numpy arrays
        and the correspoding label.
        In case of an untargeted attack the label is the ground truth label.
        In case of a targeted attack the label is the target label.
    """
    filepath = os.getenv('INPUT_YML_PATH')
    with open(filepath, 'r') as ymlfile:
        data = yaml.load(ymlfile)

    return [(key, _read_image(key), data[key]) for key in data.keys()]


def store_adversarial(file_name, adversarial):
    """
        Given the filename, stores the adversarial as .npy file.
    """
    if adversarial is not None:
        adversarial = check_image(adversarial)

    output_folder = os.getenv('OUTPUT_ADVERSARIAL_PATH')
    path = os.path.join(output_folder, file_name)
    path_without_extension = os.path.splitext(path)[0]
    np.save(path_without_extension, adversarial)
    CrowdAiNotifier.store_adversarial(file_name)


def attack_complete():
    """
        Send a notificaton to the crowd-ai backend that the attack has
        successfully completed.
    """
    CrowdAiNotifier.attack_complete()


def _wait_for_server_start(model, retried=0):
    logger.info('{0} for model server to start...'.format('Wating' if retried == 0 else 'Still waiting'))
    version = ''
    try:
        version = model.server_version()
    except RetriesExceededError:
        if 'NIPS 2018' not in version:
            if retried < 3:
                _wait_for_server_start(model, retried + 1)
            else:
                logger.error("=======> Can't reach model server: %s.", model.base_url)


def load_model():
    """
        Returns an BSONModel reading the server URI and post from
        environment variables.
    """
    model_port = os.getenv('MODEL_PORT', 8989)
    model_server = os.getenv('MODEL_SERVER', 'localhost')
    model_url = 'http://{0}:{1}'.format(model_server, model_port)
    model = TinyImageNetBSONModel(model_url)
    _wait_for_server_start(model)
    return model


def get_test_data():
    """
        Returns a list with entries (image, label) over test samples.
    """
    basepath = os.path.join(os.path.dirname(__file__), 'test_images/')
    label_file = os.path.join(basepath, 'labels.yml')
    with open(label_file, 'r') as ymlfile:
        files2labels = yaml.load(ymlfile)

    return [(_load_img(os.path.join('test_images', filename)), label)
            for filename, label in sorted(files2labels.items())]
