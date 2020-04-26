# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
from skimage.transform import resize

from hebb import HebbianNetwork


# Utils
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data


def plot(data, test, predicted, figsize=(3, 3)):
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
    plt.savefig("result_mnist.png")
    plt.show()


def pre_processing(img):
    img = resize(img, (20, 20))
    w, h = img.shape

    # Reshape
    flatten = np.reshape(img, (w * h))
    return flatten


def main():
    # Load data
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
        predicted = model.predict(data, iterations=20)
        predicted_data.append(predicted)

    print("Show prediction results...")
    plot(train_data, test_data, predicted_data, figsize=(50, 50))
    # print("Show network weights matrix...")
    # model.plot_weights()


if __name__ == '__main__':
    main()
