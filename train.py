# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('TkAgg')

import numpy as np

np.random.seed(1)
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import network
from scipy.ndimage import rotate


# Utils
def get_corrupted_input(input_data, corruption_level):
    corrupted = np.copy(input_data)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input_data))
    for i, v in enumerate(input_data):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data


def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


def rotate_img(img, angle, bg_patch=(5, 5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def pre_processing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w, h), mode='reflect')

    # Threshold
    thresh = threshold_mean(img)
    binary = img > thresh

    # Boolean to int
    shift = 2 * (binary * 1) - 1

    # Reshape
    flatten = np.reshape(shift, (w * h))
    return flatten


def main():
    # Load data
    camera = skimage.data.camera()
    astronaut = rgb2gray(skimage.data.astronaut())
    horse = skimage.data.horse()
    coffee = rgb2gray(skimage.data.coffee())

    # Marge data
    data = [camera, astronaut, horse, coffee]
    test_data = [rotate_img(d, 5) for d in data]
    test_data = [pre_processing(d) for d in test_data]

    # Pre-processing
    print("Start to data pre-processing...")
    data = [pre_processing(d) for d in data]

    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train(data)

    # Generate test set
    test = [get_corrupted_input(d, 0.0) for d in test_data]

    predicted = model.predict(test, iterations=200, threshold=70, run_async=False)
    print("Show prediction results...")
    plot(data, test, predicted)
    print("Show network weights matrix...")
    # model.plot_weights()


if __name__ == '__main__':
    main()
