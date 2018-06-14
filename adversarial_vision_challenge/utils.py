import os
import numpy as np
from PIL import Image
import csv


def _img_to_numpy(path):
    """
        Reads image image from the given path and returns an numpy array.
    """
    shape=(64, 64)
    path = os.path.join(os.path.dirname(__file__), path)
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    image = image / 255
    return image


def _read_path_and_label(input_folder, row):
    """
        Returns a tuple of image as numpy array and label as int,
        given the csv row.
    """
    file_name = row[0]
    img_path = os.path.join(input_folder, file_name)
    img = _img_to_numpy(img_path)
    true_label = row[1]
    return (file_name, img, true_label)


def read_images():
    """
        Returns a list containing tuples of images as numpy arrays
        and the correspoding true label.
    """
    input_folder = os.getenv('INPUT_CSV_PATH')
    filepath = os.path.join(input_folder, 'images.csv')
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        result = [_read_path_and_label(input_folder, row) for row in reader]
        return result

def store_adversarial(file_name, adversarial):
    """
        Given the filename, stores the adversarial as .npy file.
    """
    output_folder = os.getenv('OUTPUT_ADVERSARIAL_PATH')
    path = os.path.join(output_folder, file_name)
    path_without_extension = os.path.splitext(path)[0]
    np.save(path_without_extension, adversarial)


from adversarial_vision_challenge.utils import read_images, store_adversarial

for (file_name, img, label) in read_images():
    # run your adversarial attack
    adversarial = img # ...
    store_adversarial(file_name, adversarial)

    
