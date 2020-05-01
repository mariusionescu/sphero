# -*- coding: utf-8 -*-

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
    def __init__(
            self, layers: int,
            neurons: int,
            learning_rate: float = 0.01,
            activation_threshold: float = 0.5
    ):
        self.layers = layers
        self.neurons = neurons
        self.weights = []
        self.connections = []
        self.activation_threshold = activation_threshold
        self.connection_ratio = 0.7
        self.learning_rate = learning_rate

        for i in range(self.layers):

            # Initiate synapses
            weights = np.zeros((self.neurons, self.neurons))
            self.weights.append(weights)

            # Initiate neurons connectivity matrix
            connections = np.random.choice(
                a=[True, False],
                size=(self.neurons, self.neurons),
                p=[self.connection_ratio, 1-self.connection_ratio]
            )
            self.connections.append(connections)

    def train(self, data: np.array) -> None:

        input_data = np.copy(data)
        signals = minmax_scale(input_data)

        for idx, weights in enumerate(self.weights):

            connections = self.connections[idx]

            # Calculate propagated signals
            w = np.add(weights, signals, where=connections)
            w[w > 1] = 1

            # Calculate aggregated signals
            signals = np.sum(w, axis=1)

            # Apply activations and limits
            signals[signals < self.activation_threshold] = 0
            signals[signals > 1] = 1

            print('max energy', w)
            self.weights[idx] = w

    def predict(self, data: np.ndarray) -> np.ndarray:

        # Copy to avoid call by reference
        input_data = np.copy(data)
        signals = minmax_scale(input_data)

        for idx, weights in enumerate(self.weights):

            connections = self.connections[idx]

            # Calculate propagated signals
            w = np.add(weights, signals, where=connections)
            w[w > 1] = 1

            # Calculate aggregated signals
            signals = np.sum(w, axis=1)

            # Apply activations and limits
            signals[signals < self.activation_threshold] = 0
            signals[signals > 1] = 1

            print(signals)

        output = np.copy(signals)
        return output

    def plot_weights(self, save: bool = False):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.weights[0])
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        if save:
            plt.savefig("weights.png")
        plt.show()
