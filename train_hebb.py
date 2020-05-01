# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from skimage.transform import resize

from hebb import HebbianNetwork

PLOT_WEIGHTS = False


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data


def plot(data, test, predicted, fig_size=(3, 3), save=False):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, ax_arr = plt.subplots(len(data), 3, figsize=fig_size)
    for i in range(len(data)):
        if i == 0:
            ax_arr[i, 0].set_title('Train data')
            ax_arr[i, 1].set_title("Input data")
            ax_arr[i, 2].set_title('Output data')

        ax_arr[i, 0].imshow(data[i])
        ax_arr[i, 0].axis('off')
        ax_arr[i, 1].imshow(test[i])
        ax_arr[i, 1].axis('off')
        ax_arr[i, 2].imshow(predicted[i])
        ax_arr[i, 2].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig("predictions.png")
    plt.show()


def pre_processing(img):
    img = resize(img, (20, 20))
    w, h = img.shape

    # Reshape
    flatten = np.reshape(img, (w * h))
    return flatten


def main():

    (x_train, y_train), (_, _) = mnist.load_data()
    train_data = []
    for i in range(3):
        xi = x_train[y_train == i]
        train_data.append(xi[0])
        train_data.append(xi[1])
        train_data.append(xi[2])

    train_data = [pre_processing(d) for d in train_data]

    model = HebbianNetwork(layers=10, neurons=train_data[0].shape[0])

    # Pre processing
    print("Start to data pre processing...")
    for data in train_data:
        for i in range(100):
            model.train(data)

    # Make test datalist
    test_data = []
    for i in range(3):
        xi = x_train[y_train == i]
        test_data.append(xi[3])
        test_data.append(xi[4])
        test_data.append(xi[5])

    test_data = [pre_processing(d) for d in test_data]

    predicted_data = []

    for data in test_data:
        predicted = model.predict(data)
        predicted_data.append(predicted)

    print("Show prediction results...")
    plot(train_data, test_data, predicted_data, fig_size=(50, 50))
    if PLOT_WEIGHTS:
        print("Show network weights matrix...")
        model.plot_weights()


if __name__ == '__main__':
    main()
