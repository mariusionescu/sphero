import joblib
import logging
from base import BaseElement
from region import Region
import random
from queue import Queue


class Cortex(BaseElement):

    map = {}

    REGIONS = 1

    def __init__(self):
        super(Cortex, self).__init__()
        self.idx = 1
        self.map = {}
        self.sensors = {}
        self.queue = Queue()

        for i in range(self.REGIONS):
            region = Region(cortex=self, parent=self)
            self.children.append(region)

        self.init_synapses()

    def init_synapses(self):
        logging.info("Initialing synapses... ")
        for region in self.children:
            for layer in region.children:
                for column in layer.children:
                    for neuron in column.children:
                        neuron.init_synapses()

    def set_sensor(self, sensor):
        sensor.cortex = self
        self.sensors[sensor.name] = sensor
        while len(sensor.links) < sensor.dimension:
            layer = self.children[0].children[0]
            remote_column = random.choice(layer.children)
            remote_neuron = random.choice(remote_column.children)

            logging.info("Attaching sensor to neuron %s", remote_neuron.idx)
            sensor.links.append(remote_neuron.idx)

    def get_sensor(self, sensor_name):
        return self.sensors[sensor_name]

    @property
    def neurons(self):
        return sum([len(c.children) for r in self.children for l in r.children for c in l.children])

    @staticmethod
    def load(path):
        data = joblib.load(path)
        cortex = data['cortex']
        logging.info("Loaded %s neurons", cortex.neurons)
        logging.info("Loaded %s index items", len(cortex.map))
        cortex.queue = Queue()
        return cortex

    def save(self, path):
        logging.info("Saving %s neurons", self.neurons)
        logging.info("Saving %s index items", len(self.map))
        del self.queue
        data = {'cortex': self}
        joblib.dump(data, path)
        self.queue = Queue()


