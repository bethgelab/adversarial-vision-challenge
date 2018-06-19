import csv
import os

import numpy as np
from PIL import Image

from .client import TinyImageNetBSONModel


def _img_to_numpy(path):
    """
        Reads image image from the given path and returns an numpy array.
    """
    shape = (64, 64)
    path = os.path.join(os.path.dirname(__file__), path)
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    image = image / 255
    return image


def _read_path_and_label(row):
    """
        Returns a tuple of image as numpy array and label as int,
        given the csv row.
    """
    input_folder = os.getenv('INPUT_IMG_PATH')
    file_name = row[0]
    img_path = os.path.join(input_folder, file_name)
    img = _img_to_numpy(img_path)
    true_label = np.int64(row[1])
    return (file_name, img, true_label)


def read_images():
    """
        Returns a list containing tuples of images as numpy arrays
        and the correspoding true label.
    """
    filepath = os.getenv('INPUT_CSV_PATH')
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        result = [_read_path_and_label(row) for row in reader]
        return result


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
