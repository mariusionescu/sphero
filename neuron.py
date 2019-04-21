from base import BaseElement
import logging
import random
logging.basicConfig(level=logging.DEBUG)


class Neuron(BaseElement):

    def __init__(self, cortex, parent, n_children):
        super(Neuron, self).__init__(cortex, parent, n_children)

        self.set_idx()

    def fire(self):
        pass

    def predict(self):
        pass

    def init_synapses(self):
        while len(self.links) < self.n_children:
            self.connect()

        logging.info("[NEURON %s] %s synapses have been initialized", self.idx, len(self.links))

    def connect(self):
        remote_column = random.choice(self.parent.parent.children)
        remote_neuron = random.choice(remote_column.children)

        if self.idx not in remote_neuron.links:
            self.links[remote_neuron.idx] = 0.0
