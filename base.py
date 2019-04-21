import logging
logging.basicConfig(level=logging.DEBUG)


class BaseElement(object):

    def __init__(self, cortex, parent, n_children):
        self.idx = None
        self.cortex = cortex
        self.parent = parent
        self.n_children = n_children
        self.children = []
        self.links = {}

    @property
    def next_child_id(self):
        return len(self.children) + 1

    def set_idx(self):
        if self.idx:
            return
        if type(self.parent.idx) == int:
            self.idx = (self.parent.idx, self.parent.next_child_id)
        else:
            self.idx = self.parent.idx + (self.parent.next_child_id,)
        logging.info("Allocating id %s", self.idx)
        self.cortex.map[self.idx] = self
        return self.idx
