import foolbox
from adversarial_vision_challenge import model_server, read_images, store_adversarial
import os
import tensorflow as tf
import numpy as np
import requests
import threading
import pytest


class ServerThread(object):
    def __init__(self, model):
        self.__model = model
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        model_server(self.__model)


def create_test_model():
    images = tf.placeholder(tf.float32, (None, 64, 64, 3))
    # to grayscale
    net = tf.reduce_mean(images, axis=3)
    # flatten spatial axes
    net = tf.layers.flatten(images)
    # just use the first 200 pixels as logits for this test network
    logits = net[:, :200]
    return images, logits


def run_attack(model, image, label):
    attack = foolbox.attacks.AdditiveGaussianNoiseAttack()
    criterion = foolbox.criteria.Misclassification()
    adversarial = foolbox.Adversarial(model, criterion, image, label)
    attack(adversarial)
    print(adversarial.distance.value)
    return adversarial.image


def cleanup(adversarials_path):
    for file in os.listdir(adversarials_path):
        path = os.path.join(adversarials_path, file)
        os.remove(path)


def fixture():
    cwd = os.getcwd()
    print(cwd)
    adversarials_path = os.path.join(cwd, 'ci', 'adversarials')
    os.environ['INPUT_YML_PATH'] = os.path.join(cwd, 'ci', 'images.yml')
    os.environ['INPUT_IMG_PATH'] = os.path.join(cwd, 'ci')
    os.environ['OUTPUT_ADVERSARIAL_PATH'] = adversarials_path
    images, logits = create_test_model()
    fmodel = foolbox.models.TensorFlowModel(images, logits, (0, 255))
    ServerThread(fmodel)
    return adversarials_path, cwd, fmodel


def stop_server():
    requests.get('http://localhost:8989/shutdown')


@pytest.mark.filterwarnings("ignore:inspect.getargspec")
def test_untargeted_attack():
    adversarials_path, cwd, fmodel = fixture()

    for (file_name, image, label) in read_images():
        print('predicted model for: ', file_name, np.argmax(fmodel.predictions(image)))
        adversarial = run_attack(fmodel, image, label)
        store_adversarial(file_name, adversarial)

    stop_server()

    for i in range(9):
        adversarial = os.path.join(cwd, 'ci', 'val_{}.npy'.format(i))
        array = np.load(adversarial)

        assert array.shape == (64, 64, 3)

    cleanup(adversarials_path)

    print("DONE.")
