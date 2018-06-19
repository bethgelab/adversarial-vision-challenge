#!/usr/bin/env python3
import foolbox
import tensorflow as tf
from adversarial_vision_challenge import model_server


def create_test_model():
    images = tf.placeholder(tf.float32, (None, 64, 64, 3))
    # to grayscale
    net = tf.reduce_mean(images, axis=3)
    # flatten spatial axes
    net = tf.layers.flatten(images)
    # just use the first 200 pixels as logits for this test network
    logits = net[:, :200]
    return images, logits


def main():
    images, logits = create_test_model()
    fmodel = foolbox.models.TensorFlowModel(images, logits, (0, 255))
    model_server(fmodel)


if __name__ == '__main__':
    main()
