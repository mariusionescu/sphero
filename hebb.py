# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import typing
import settings
import ipdb

log = settings.get_logger()

np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f},'.format})


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
            decay_rate: float = 0.1,
            activation_threshold: float = 0.7
    ):
        self.layers = layers
        self.neurons = neurons
        self.weights = []
        self.connections = []
        self.activation_threshold = activation_threshold
        self.connection_ratio = 0.1
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        log.info(
            'settings',
            layers=self.layers,
            neurons=self.neurons,
            activation_threshold=self.activation_threshold,
            connection_ratio=self.connection_ratio,
            learning_rate=self.learning_rate
        )

        for layer in range(self.layers):
            log.info('weights.init', layer=layer)

            weights = np.zeros((self.neurons, self.neurons), dtype=np.float16)
            self.weights.append(weights)

            # Initiate neurons connectivity matrix
            connections = np.random.choice(
                a=[True, False],
                size=(self.neurons, self.neurons),
                p=[self.connection_ratio, 1-self.connection_ratio],
            )
            self.connections.append(connections)

    @staticmethod
    def normalize(array):
        array[np.isnan(array)] = 0.0
        array[array > 1.0] = 1.0
        return array

    def train(self, data: np.array) -> None:

        input_data = np.array(data, dtype=np.float16)
        input_signals = minmax_scale(input_data)

        aggregated_signals = None

        for layer, weights in enumerate(self.weights):

            if aggregated_signals is None:
                aggregated_signals = input_signals

            log.info('train', layer=layer)

            bias_array = np.random.randint(8, 10, size=self.neurons) * self.decay_rate
            bias_array = np.array(bias_array, np.float16)
            aggregated_signals = aggregated_signals * bias_array
            self.normalize(aggregated_signals)
            # log.info('signals', type='aggregated', values=aggregated_signals)

            connections = self.connections[layer]

            # Calculate propagated signals
            rates = np.zeros((self.neurons, self.neurons))
            rates = np.add(rates, 1.0, where=connections)
            rates = np.multiply(rates, aggregated_signals, where=connections)
            rates[rates > 0] = self.learning_rate
            new_weights = np.add(weights, rates, where=connections)
            self.normalize(new_weights)

            propagated_signals = np.multiply(new_weights, aggregated_signals, where=connections)
            self.normalize(propagated_signals)

            # Calculate aggregated signals
            aggregated_signals = np.sum(propagated_signals, axis=0)
            # ipdb.set_trace()

            # Apply activations and limits
            aggregated_signals[aggregated_signals < self.activation_threshold] = 0.0

            self.weights[layer] = new_weights

    def predict(self, data: np.ndarray) -> np.ndarray:

        # Copy to avoid call by reference
        input_data = np.copy(data)
        input_signals = minmax_scale(input_data)

        aggregated_signals = None

        for layer, weights in enumerate(self.weights):

            if aggregated_signals is None:
                aggregated_signals = input_signals

            bias_array = np.random.randint(8, 10, size=self.neurons) * self.decay_rate
            aggregated_signals = aggregated_signals * bias_array
            self.normalize(aggregated_signals)

            connections = self.connections[layer]

            # Calculate propagated signals
            propagated_signals = np.multiply(weights, aggregated_signals, where=connections)
            self.normalize(propagated_signals)

            # Calculate aggregated signals
            aggregated_signals = np.sum(propagated_signals, axis=0)

            # Apply activations and limits
            aggregated_signals[aggregated_signals < self.activation_threshold] = 0.0

        output = np.copy(aggregated_signals)
        self.normalize(output)
        return output

    def plot_weights(self, layer: int = -1, save: bool = False):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.weights[layer])
        plt.clim(0, 1)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        if save:
            plt.savefig("weights.png")
        plt.show()
