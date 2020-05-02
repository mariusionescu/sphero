# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import typing
import ipdb


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
            learning_rate: float = 0.001,
            activation_threshold: float = 0.9
    ):
        self.layers = layers
        self.neurons = neurons
        self.weights = []
        self.connections = []
        self.activation_threshold = activation_threshold
        self.connection_ratio = 0.01
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
        aggregated_signals = minmax_scale(input_data)

        aggregated_signals /= self.neurons

        # Add bias
        bias_array = np.random.randint(8, 10, size=self.neurons) * 0.1
        aggregated_signals = aggregated_signals * bias_array

        for idx, weights in enumerate(self.weights):

            connections = self.connections[idx]

            # print('aggregated_signals', aggregated_signals)

            # Calculate propagated signals
            propagated_signals = np.add(weights, aggregated_signals, where=connections)
            propagated_signals[np.isnan(propagated_signals)] = 0
            propagated_signals[propagated_signals > 1] = 1

            # Calculate aggregated signals
            aggregated_signals = np.sum(propagated_signals, axis=0)

            # Apply activations and limits
            aggregated_signals[aggregated_signals < self.activation_threshold] = 0.0
            aggregated_signals[aggregated_signals > 0.0] = self.learning_rate

            # print('aggregated_signals', aggregated_signals)

            self.weights[idx] = propagated_signals

    def predict(self, data: np.ndarray) -> np.ndarray:

        # Copy to avoid call by reference
        input_data = np.copy(data)
        aggregated_signals = minmax_scale(input_data)

        for idx, weights in enumerate(self.weights):

            # print('weights', weights)
            connections = self.connections[idx]

            # Calculate propagated signals
            propagated_signals = np.add(weights, aggregated_signals, where=connections)
            propagated_signals[np.isnan(propagated_signals)] = 0
            propagated_signals[propagated_signals > 1.0] = 1.0

            # print('propagated_signals', propagated_signals)

            # Calculate aggregated signals
            aggregated_signals = np.sum(propagated_signals, axis=0)
            # print('aggregated_signals', aggregated_signals)

            # Apply activations and limits
            aggregated_signals[aggregated_signals < self.activation_threshold] = 0.0
            aggregated_signals[aggregated_signals > 1.0] = 1.0

        output = np.copy(aggregated_signals)
        # print(output)
        return output

    def plot_weights(self, layer: int = -1, save: bool = False):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.weights[layer])
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        if save:
            plt.savefig("weights.png")
        plt.show()
