# -*- coding: utf-8 -*-

from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import typing


class HopfieldNetwork(object):
    def __init__(self):
        self.neuron_count = None
        self.weights = None
        self.iterations = None
        self.threshold = None
        self.run_async = False

    def train(self, train_data: typing.List[np.array]) -> None:
        print("Start to train weights...")
        num_data = len(train_data)
        self.neuron_count = train_data[0].shape[0]

        print('NEURON COUNT', self.neuron_count)

        # initialize weights
        weights = np.zeros((self.neuron_count, self.neuron_count))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.neuron_count)

        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            weights += np.outer(t, t)

        # Make diagonal element of W into 0
        diag_w = np.diag(np.diag(weights))
        weights = weights - diag_w
        weights /= num_data

        self.weights = weights

    def predict(self,
                data: typing.List[np.ndarray],
                iterations: int = 20,
                threshold: int = 30,
                run_async: bool = False) -> typing.List[np.ndarray]:
        print("Start to predict...")
        self.iterations = iterations
        self.threshold = threshold
        self.run_async = run_async

        # Copy to avoid call by reference 
        copied_data = np.copy(data)

        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted

    def _run(self, init_s) -> np.ndarray:
        if not self.run_async:
            """
            Synchronous update
            """
            # Compute initial state energy
            s = init_s
            e = self.energy(s)

            # Iteration
            for i in range(self.iterations):
                # Update s
                s = np.sign(self.weights @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)

                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
        else:
            """
            Asynchronous update
            """
            # Compute initial state energy
            s = init_s
            e = self.energy(s)

            # Iteration
            for i in range(self.iterations):
                rand_count = int(self.neuron_count * 0.1)
                for j in range(rand_count):
                    # Select random neuron
                    idx = np.random.randint(0, self.neuron_count)
                    # Update s
                    s[idx] = np.sign(self.weights[idx].T @ s - self.threshold)

                # Compute new state energy
                e_new = self.energy(s)

                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s

    def energy(self, s: np.ndarray) -> float:
        return -0.5 * s @ self.weights @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.weights, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()
