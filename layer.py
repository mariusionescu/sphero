from base import BaseElement
from column import Column


class Layer1(BaseElement):

    COLUMNS = 1000
    LOCAL_SYNAPSES = 0
    REMOTE_SYNAPSES = 10
    FORWARD_SYNAPSES = 1

    def __init__(self, cortex, parent):
        super(Layer1, self).__init__(cortex, parent)
        self.activated_neurons = set()
        self.set_idx()

        for i in range(self.COLUMNS):
            column = Column(cortex, parent=self)
            self.children.append(column)


class Layer2(BaseElement):

    COLUMNS = 100
    LOCAL_SYNAPSES = 1
    REMOTE_SYNAPSES = 2
    FORWARD_SYNAPSES = 1

    def __init__(self, cortex, parent):
        super(Layer2, self).__init__(cortex, parent)
        self.activated_neurons = set()
        self.set_idx()

        for i in range(self.COLUMNS):
            column = Column(cortex, parent=self)
            self.children.append(column)


class Layer3(BaseElement):

    COLUMNS = 50
    LOCAL_SYNAPSES = 1
    REMOTE_SYNAPSES = 1
    FORWARD_SYNAPSES = 0

    def __init__(self, cortex, parent):
        super(Layer3, self).__init__(cortex, parent)
        self.activated_neurons = set()
        self.set_idx()

        for i in range(self.COLUMNS):
            column = Column(cortex, parent=self)
            self.children.append(column)