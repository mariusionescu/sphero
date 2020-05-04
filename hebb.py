# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import typing
import settings
import ipdb

log = settings.get_logger()

np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})


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
            learning_rate: float = 0.1,
            decay_rate: float = 0.1,
            activation_threshold: float = 0.5
    ):
        self.layers = layers
        self.neurons = neurons
        self.weights = []
        self.connections = []
        self.activation_threshold = activation_threshold
        self.connection_ratio = 0.6
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        log.info(
            'network.settings',
            layers=self.layers,
            neurons=self.neurons,
            activation_threshold=self.activation_threshold,
            connection_ratio=self.connection_ratio,
            learning_rate=self.learning_rate
        )

        for layer in range(self.layers):

            # Initiate neurons connectivity matrix
            connections = np.random.choice(
                a=[True, False],
                size=(self.neurons, self.neurons),
                p=[self.connection_ratio, 1-self.connection_ratio],
            )
            self.connections.append(connections)

            weights = np.random.uniform(0.001, 0.002, (self.neurons, self.neurons))
            weights = np.multiply(weights, 0, where=~connections, dtype=np.float32)
            self.weights.append(weights)

        log.info('layers.init', layers=self.layers)

    @staticmethod
    def normalize(array):
        array[np.isnan(array)] = 0.0
        array[array > 1.0] = 1.0
        return array

    def update_weights(self, input_array, layer):
        weights = self.weights[layer]
        connections = self.connections[layer]

        # Calculate rates
        rates = np.zeros((self.neurons, self.neurons))
        rates = np.add(rates, 1, where=connections, dtype=np.float32)
        # rates[rates < 1] = 0
        # print('RATES_0', rates)
        # print('INPUT', input_array)
        rates = np.multiply(rates, input_array, where=connections, dtype=np.float32)
        # print('RATES_1', rates)
        rates[rates < self.activation_threshold] = 0
        rates = rates * self.learning_rate

        # Calculates new weights
        new_weights = np.add(weights, rates, where=connections)
        self.normalize(new_weights)
        self.weights[layer] = new_weights
        print('new_weights', new_weights)

    def propagate(self, input_array, layer):

        weights = self.weights[layer]
        connections = self.connections[layer]

        # print('propagate_last_output_array_0', input_array)
        # Add bias and normalize input
        bias_array = np.random.randint(8, 10, size=self.neurons) * self.decay_rate
        input_array = input_array * bias_array
        # print('propagate_last_output_array_1', input_array)
        self.normalize(input_array)
        # print('propagate_last_output_array_2', input_array)

        # Calculate propagated signals
        # propagated_signals = np.multiply(weights, input_array, where=connections, dtype=np.float32)
        # print('propagated_signals', propagated_signals)
        # output_array = np.average(propagated_signals, axis=1)
        output_array = np.dot(weights, input_array)
        self.normalize(output_array)

        # Calculate aggregated signals

        # Apply activations and limits
        # output_array[output_array < self.activation_threshold] = 0.0

        self.normalize(output_array)
        return output_array

    def train(self, data: np.array) -> None:

        input_array = np.copy(data)
        input_array = minmax_scale(input_array)
        last_output_array = None

        print('input_array', input_array)
        for layer in range(self.layers):
            if last_output_array is None:
                last_output_array = input_array

            print('last_output_array', layer, last_output_array)
            last_output_array = self.propagate(last_output_array, layer)
            self.update_weights(last_output_array, layer)

    def predict(self, data: np.ndarray) -> np.ndarray:

        input_array = np.copy(data)
        input_array = minmax_scale(input_array)
        last_output_array = None

        for layer in range(self.layers):
            if last_output_array is None:
                last_output_array = input_array

            last_output_array = self.propagate(last_output_array, layer)
        return last_output_array

    def plot(self, save: bool = False):

        fig, ax_arr = plt.subplots(len(self.weights), 2, figsize=(10, 10))

        if len(self.weights) == 1:
            ax_arr.set_title('Layer 0')
            ax_arr.imshow(self.weights[0])
            ax_arr.axis('off')
        else:
            for i in range(len(self.weights)):
                ax_arr[i, 0].set_title('Layer {}'.format(i))
                img = ax_arr[i, 0].imshow(self.weights[i], vmin=0, vmax=1)
                ax_arr[i, 0].axis('off')

                fig.colorbar(img, ax_arr[i, 1])

        if save:
            plt.savefig("predictions.png")

        plt.show()
