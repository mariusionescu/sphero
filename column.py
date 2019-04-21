from base import BaseElement
from neuron import Neuron


class Column(BaseElement):

    def __init__(self, cortex, parent, n_children):
        super(Column, self).__init__(cortex, parent, n_children)

        self.set_idx()

        for i in range(n_children):
            neuron = Neuron(cortex, parent=self, n_children=100)
            self.children.append(neuron)