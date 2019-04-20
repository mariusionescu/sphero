import joblib
import logging
import random

logging.basicConfig(level=logging.DEBUG)

index = {}


class BaseElement(object):

    def __init__(self, *args, **kwargs):
        self.idx = None
        self.parent = None
        self.children = []
        self.links = []

    @property
    def next_child_id(self):
        return len(self.children) + 1

    def set_idx(self):
        if self.idx:
            return
        self.idx = self.parent.idx, self.parent.next_child_id
        logging.info("Allocating id %s", self.idx)
        index[self.idx] = self
        return self.idx


class Neuron(BaseElement):

    def __init__(self, parent, n_synapses):
        super(Neuron, self).__init__()
        self.parent = parent
        self.n_synapses = n_synapses
        self.column = self.parent
        self.synapses = self.links

        self.set_idx()

    def fire(self):
        pass

    def predict(self):
        pass

    def init_synapses(self):
        while len(self.synapses) < self.n_synapses:
            self.connect()

        logging.info("[NEURON %s] %s synapses have been initialized", self.idx, self.n_synapses)

    def connect(self):
        remote_column = random.choice(self.parent.parent.columns)
        remote_neuron = random.choice(remote_column.neurons)

        if self.idx not in remote_neuron.links:
            self.links.append(remote_neuron.idx)


class Column(BaseElement):

    def __init__(self, parent, n_neurons):
        super(Column, self).__init__()
        self.parent = parent
        self.n_neurons = n_neurons
        self.zone = self.parent
        self.neurons = self.children

        self.set_idx()

        for i in range(self.n_neurons):
            neuron = Neuron(parent=self, n_synapses=100)
            self.neurons.append(neuron)


class Layer(BaseElement):

    def __init__(self, parent, n_columns):
        super(Layer, self).__init__()
        self.parent = parent
        self.n_columns = n_columns
        self.columns = self.children

        self.set_idx()

        for i in range(n_columns):
            column = Column(parent=self, n_neurons=10)
            self.columns.append(column)


class Region(BaseElement):

    def __init__(self, parent, n_layers):
        super(Region, self).__init__()
        self.parent = parent
        self.n_layers = n_layers
        self.cortex = self.parent
        self.layers = self.children

        self.set_idx()

        for i in range(n_layers):
            layer = Layer(parent=self, n_columns=10000)
            self.layers.append(layer)


class Cortex(BaseElement):

    def __init__(self, n_zones):
        super(Cortex, self).__init__()
        self.idx = 1
        self.regions = self.children
        self.sensors = {}

        for i in range(n_zones):
            region = Region(parent=self, n_layers=3)
            self.regions.append(region)

        self.init_synapses()

    def init_synapses(self):
        logging.info("Initialing synapses... ")
        for region in self.regions:
            for layer in region.layers:
                for column in layer.columns:
                    for neuron in column.neurons:
                        neuron.init_synapses()

    def set_sensor(self, sensor_obj):
        self.sensors[sensor_obj.name] = sensor_obj

    def get_sensor(self, sensor_name):
        return self.sensors[sensor_name]

    @property
    def neurons(self):
        return sum([len(c.neurons) for r in self.regions for l in r.layers for c in l.columns])

    @staticmethod
    def load(path):
        global index
        data = joblib.load(path)
        index = data['index']
        cortex = data['cortex']
        logging.info("Loaded %s neurons", cortex.neurons)
        logging.info("Loaded %s index items", len(index))
        return cortex

    def save(self, path):
        logging.info("Saving %s neurons", self.neurons)
        logging.info("Saving %s index items", len(index))
        data = {'cortex': self, 'index': index}
        joblib.dump(data, path)


