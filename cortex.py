import joblib
import logging
from base import BaseElement
from region import Region

logging.basicConfig(level=logging.DEBUG)


class Cortex(BaseElement):

    map = {}

    def __init__(self, cortex, parent, n_children):
        super(Cortex, self).__init__(cortex, parent, n_children)
        self.idx = 1
        self.map = {}
        self.sensors = {}

        for i in range(n_children):
            region = Region(cortex=self, parent=self, n_children=3)
            self.children.append(region)

        self.init_synapses()

    def init_synapses(self):
        logging.info("Initialing synapses... ")
        for region in self.children:
            for layer in region.children:
                for column in layer.children:
                    for neuron in column.children:
                        neuron.init_synapses()

    def set_sensor(self, sensor_obj):
        self.sensors[sensor_obj.name] = sensor_obj

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
        return cortex

    def save(self, path):
        logging.info("Saving %s neurons", self.neurons)
        logging.info("Saving %s index items", len(self.map))
        data = {'cortex': self}
        joblib.dump(data, path)


