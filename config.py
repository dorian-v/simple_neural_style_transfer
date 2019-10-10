import numpy as np

class CONFIG:
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 360
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'  # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    OUTPUT_DIR = 'output/'