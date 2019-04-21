from base import BaseElement
from column import Column


class Layer(BaseElement):

    def __init__(self, cortex, parent, n_children):
        super(Layer, self).__init__(cortex, parent, n_children)

        self.set_idx()

        for i in range(n_children):
            column = Column(cortex, parent=self, n_children=10)
            self.children.append(column)
