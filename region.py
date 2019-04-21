from base import BaseElement
from layer import Layer


class Region(BaseElement):

    def __init__(self, cortex, parent, n_children):
        super(Region, self).__init__(cortex, parent, n_children)

        self.set_idx()

        for i in range(n_children):
            layer = Layer(cortex, parent=self, n_children=1000)
            self.children.append(layer)
