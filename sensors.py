import numpy as np
import logging


class BaseSensor(object):
    def __init__(self, name, cortex=None, dimension=100):
        self.name = name
        self.dimension = dimension * 11
        self.links = []
        self.cortex = cortex

    @staticmethod
    def scale(array, min_limit=0.6, max_limit=0.9, output=int):
        array = np.array(array, dtype=float)
        array += -(np.min(array))
        array /= np.max(array) / (max_limit - min_limit)
        array += min_limit
        array = array.tolist()
        if output in (int, 'int'):
            array = list(map(int, array))
        return array

    @staticmethod
    def to_bin(array):
        return ''.join([str(bin(i))[2:].rjust(11, '0') for i in array])


class Vision(BaseSensor):
    def send(self, data):
        data = self.scale(data)


class Sound(BaseSensor):
    def send(self, data):
        data = self.scale(data)


class Text(BaseSensor):
    def send(self, data):
        data = self.scale(data)


class Array(BaseSensor):
    async def send(self, data):
        data = self.scale(data, 0, 1024, output=int)
        data = self.to_bin(data)
        logging.info("[SENSOR] sending data %s to sensor %s", data, self.name)

        for i, neuron_activation in enumerate(data):

            if neuron_activation == '1':
                neuron_idx = self.links[i]
                neuron = self.cortex.map[neuron_idx]
                logging.info("[SENSOR] firing %s -> %s", i, neuron_idx)
                await neuron.fire(remote_idx=i, strength=0.9)
