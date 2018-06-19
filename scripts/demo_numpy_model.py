#!/usr/bin/env python3
from adversarial_vision_challenge import model_server


class Model(object):
    def channel_axis(self):
        return 1

    def bounds(self):
        return (0, 255)

    def predictions(self, image):
        return 22


def main():
    model = Model()
    model_server(model)


if __name__ == '__main__':
    main()
