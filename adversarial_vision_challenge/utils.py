import csv
import os

import numpy as np
import yaml

from .client import TinyImageNetBSONModel


def _img_to_numpy(path):
    """
        Reads image image from the given path and returns an numpy array.
    """
    shape = (64, 64)
    path = os.path.join(os.path.dirname(__file__), path)
    image = np.load(path)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    image = image / 255
    return image

def _read_image(file_name):
    """
        Returns a tuple of image as numpy array and label as int,
        given the csv row.
    """
    input_folder = os.getenv('INPUT_IMG_PATH')
    img_path = os.path.join(input_folder, file_name)
    return _img_to_numpy(img_path)

def read_images():
    """
        Returns a list containing tuples of images as numpy arrays
        and the correspoding true label.
    """
    filepath = os.getenv('INPUT_YML_PATH')
    with open(filepath, 'r') as ymlfile:
        data = yaml.load(ymlfile)

    return [(key, _read_image(key), data[key]) for key in data.keys()]


def store_adversarial(file_name, adversarial):
    """
        Given the filename, stores the adversarial as .npy file.
    """
    output_folder = os.getenv('OUTPUT_ADVERSARIAL_PATH')
    path = os.path.join(output_folder, file_name)
    path_without_extension = os.path.splitext(path)[0]
    np.save(path_without_extension, adversarial)


def load_model():
    """
        Returns an BSONModel reading the server URI and post from
        environment variables.
    """
    model_port = os.getenv('MODEL_PORT', 8989)
    model_server = os.getenv('MODEL_SERVER', 'localhost')
    model_url = 'http://{0}:{1}'.format(model_server, model_port)
    model = TinyImageNetBSONModel(model_url)
    return model

def get_test_data():
    """
        Returns a list with entries (image, label) over test samples.
    """
    basepath = os.path.join(os.path.dirname(__file__), 'test_images/')
    label_file = os.path.join(basepath, 'labels.yml')
    with open(label_file, 'r') as ymlfile:
        labels = yaml.load(ymlfile)

    return [(np.load(os.path.join(basepath, key)), labels[key]) for key in labels.keys()]
