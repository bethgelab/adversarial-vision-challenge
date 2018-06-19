#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
from adversarial_vision_challenge import load_model


def main():
    model = load_model()
    np.random.seed(22)
    test_image = np.random.uniform(size=(64, 64, 3)).astype(np.float32) * 255
    prediction = model(test_image)
    print('predicted class: {}'.format(prediction))


if __name__ == '__main__':
    main()
