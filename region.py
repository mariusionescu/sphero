from base import BaseElement
from layer import Layer1, Layer2, Layer3


class Region(BaseElement):

    def __init__(self, cortex, parent):
        super(Region, self).__init__(cortex, parent)
        self.set_idx()

        layer1 = Layer1(cortex, parent=self)
        self.children.append(layer1)

        layer2 = Layer2(cortex, parent=self)
        self.children.append(layer2)

        layer3 = Layer3(cortex, parent=self)
        self.children.append(layer3)
