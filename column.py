from base import BaseElement
from neuron import Neuron


class Column(BaseElement):

    NEURONS = 10

    def __init__(self, cortex, parent):
        super(Column, self).__init__(cortex, parent)
        self.set_idx()

        for i in range(self.NEURONS):
            neuron = Neuron(cortex, parent=self)
            self.children.append(neuron)