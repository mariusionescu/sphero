# -*- coding: utf-8 -*-

from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import typing


def connectivity_matrix(a):
    a = np.array(a)
    b = np.zeros(shape=(a.shape[0], a.shape[0]))
    upper = np.triu(b + a)
    lower = np.tril(np.transpose(b + a))
    d = (upper + lower) * (np.full(a.shape[0], fill_value=1) - np.eye(a.shape[0]))
    return d


class HebbianNetwork(object):
    def __init__(self, layers: int, neurons: int, threshold: float = 0.5):
        self.layers = layers
        self.neurons = neurons
        self.weights = []
        self.threshold = threshold

        for i in range(self.layers):
            weights = np.zeros((self.neurons, self.neurons))
            self.weights.append(weights)

    def train(self, data: np.array) -> None:

        input_data = np.copy(data)
        input_data = minmax_scale(input_data)

        for idx, weights in enumerate(self.weights):

            input_connections = connectivity_matrix(input_data)

            w = weights + input_connections
            w = minmax_scale(w)
            # w[w <= self.threshold] = 0

            n = np.sum(w, axis=0)
            n = minmax_scale(n)
            n[n <= self.threshold] = 0

            signal_connections = connectivity_matrix(n)

            weights += signal_connections
            minmax_scale(weights, copy=False)

            # Make diagonal element of W into 0
            diag_w = np.diag(np.diag(weights))
            weights = weights - diag_w

            input_data = np.sum(weights, axis=0)
            # input_data = minmax_scale(input_data)
            input_data[input_data <= self.threshold] = 0

            self.weights[idx] = weights

    def predict(self,
                data: np.ndarray,
                iterations: int = 20) -> np.ndarray:

        # Copy to avoid call by reference
        input_data = np.copy(data)
        input_data = minmax_scale(input_data)

        for weights in self.weights:
            input_connections = connectivity_matrix(input_data)

            w = weights + input_connections
            # w = minmax_scale(w)

            input_data = np.sum(w, axis=0)
            input_data = minmax_scale(input_data)
            input_data[input_data <= self.threshold] = 0

        return input_data

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.weights[0])
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()
