from base import BaseElement
import logging
import random


class Neuron(BaseElement):

    LEARNING_RATE = 0.1
    FORWARD_LOSS_RATE = 0.3
    PASSIVE_LOSS_RATE = 0.2
    FORWARD_ACTIVATION_THRESHOLD = 0.4
    PASSIVE_ACTIVATION_THRESHOLD = 0.3
    PASSIVE_ACTIVATIONS = 2

    def __init__(self, cortex, parent):
        super(Neuron, self).__init__(cortex, parent)

        self.local_links = {}
        self.remote_links = {}
        self.forward_links = {}

        self.back_local_links = {}
        self.back_remote_links = {}
        self.back_forward_links = {}

        self.link_types = {
            'local': self.local_links,
            'remote': self.remote_links,
            'forward': self.forward_links
        }

        self.back_link_types = {
            'local': self.back_local_links,
            'remote': self.back_remote_links,
            'forward': self.back_forward_links
        }

        self.set_idx()

    def backward_optimization(self, idx, learning_rate, link_type):
        links = self.link_types[link_type]

        strength = links[idx]

        if learning_rate < 0:
            new_strength = max(0.0, strength + learning_rate)
            links[idx] = new_strength
            logging.debug("[BACKWARD|FORGET] adjusted link %s -> %s (%.2f)",
                          self.idx, idx, links[idx])
        else:
            new_strength = min(0.99, strength + learning_rate)
            links[idx] = new_strength
            logging.debug("[BACKWARD|LEARNING] adjusted link %s -> %s (%.2f)",
                          self.idx, idx, links[idx])

    def fire_next(self, strength=0.0, remote_idx=None, link_type='local'):

        if strength > self.FORWARD_ACTIVATION_THRESHOLD:
            self.parent.parent.activated_neurons.add(self.idx)

            neuron = self.get_neuron(remote_idx)

            new_strength = max(0.0, strength - self.FORWARD_LOSS_RATE)

            future = neuron.fire(strength=new_strength, remote_idx=self.idx, link_type=link_type)
            logging.debug("[FUTURE|ACTIVE] queueing %s -> %s (%.2f)",
                          self.idx, remote_idx, new_strength)
            self.cortex.queue.put(future)

        if strength > self.PASSIVE_ACTIVATION_THRESHOLD:

            self.parent.parent.activated_neurons.add(self.idx)

            neuron = self.get_neuron(remote_idx)

            new_strength = max(0.0, strength - self.PASSIVE_LOSS_RATE)

            future = neuron.fire(strength=new_strength, remote_idx=self.idx, link_type=link_type)
            logging.debug("[FUTURE|PASSIVE] queueing %s -> %s (%.2f)",
                          self.idx, remote_idx, new_strength)
            self.cortex.queue.put(future)

    async def fire(self, strength=0.0, remote_idx=None, link_type='local'):

        logging.info("[FIRE|RECEIVING] %s -> %s", self.idx, strength)

        if type(remote_idx) is tuple:
            links = self.link_types[link_type]
            links[remote_idx] = strength

        for neuron_idx, next_strength in self.local_links.items():
            next_strength = max(next_strength, strength)
            self.fire_next(strength=next_strength, remote_idx=neuron_idx, link_type='local')
            for prev_neuron_idx, strength in self.back_local_links.items():
                if type(prev_neuron_idx) is not tuple:
                    continue
                if strength > 0.0 and neuron_idx != remote_idx:
                    neuron = self.get_neuron(prev_neuron_idx)
                    neuron.backward_optimization(self.idx, self.LEARNING_RATE, 'local')

        for neuron_idx, next_strength in self.remote_links.items():
            next_strength = max(next_strength, strength)
            self.fire_next(strength=next_strength, remote_idx=neuron_idx, link_type='remote')
            for prev_neuron_idx, strength in self.back_remote_links.items():
                if type(prev_neuron_idx) is not tuple:
                    continue
                if strength > 0.0 and neuron_idx != remote_idx:
                    neuron = self.get_neuron(prev_neuron_idx)
                    neuron.backward_optimization(self.idx, self.LEARNING_RATE, 'remote')

        for neuron_idx, next_strength in self.forward_links.items():
            next_strength = max(next_strength, strength)
            self.fire_next(strength=next_strength, remote_idx=neuron_idx, link_type='forward')
            for prev_neuron_idx, strength in self.back_forward_links.items():
                if type(prev_neuron_idx) is not tuple:
                    continue
                if strength > 0.0 and neuron_idx != remote_idx:
                    neuron = self.get_neuron(prev_neuron_idx)
                    neuron.backward_optimization(self.idx, self.LEARNING_RATE, 'forward')

    def predict(self):
        pass

    def init_synapses(self):

        local_connections = 0
        for _ in range(self.parent.parent.LOCAL_SYNAPSES):
            if self.local_connect():
                local_connections += 1
        logging.info("[NEURON %s] initialized %s LOCAL_SYNAPSES", self.idx, local_connections)

        remote_connections = 0
        for _ in range(self.parent.parent.REMOTE_SYNAPSES):
            if self.remote_connect():
                remote_connections += 1
        logging.info("[NEURON %s] initialized %s REMOTE_SYNAPSES", self.idx, remote_connections)

        forward_connections = 0
        for _ in range(self.parent.parent.FORWARD_SYNAPSES):
            if self.forward_connect():
                forward_connections += 1
        logging.info("[NEURON %s] initialized %s FORWARD_SYNAPSES", self.idx, forward_connections)

    def try_connection(self, remote_idx, link_type):
        if (
                remote_idx != self.idx and
                remote_idx not in self.local_links and
                remote_idx not in self.remote_links and
                remote_idx not in self.forward_links
        ):
            links = self.link_types[link_type]
            links[remote_idx] = 0.0
            return self.idx
        else:
            logging.info("[NEURON %s] already has a connection to %s", self.idx, remote_idx)

    def local_connect(self):
        remote_neuron = random.choice(self.parent.children)
        remote_idx = remote_neuron.try_connection(self.idx, 'local')

        if remote_idx:
            logging.info("[CONNECTION|LOCAL] %s -> %s", self.idx, remote_idx)
            self.local_links[remote_idx] = 0.0
            return remote_idx

    def remote_connect(self):
        remote_column = random.choice(self.parent.parent.children)
        remote_neuron = random.choice(remote_column.children)

        remote_idx = remote_neuron.try_connection(self.idx, 'remote')

        if remote_idx:
            logging.info("[CONNECTION|REMOTE] %s -> %s", self.idx, remote_idx)
            self.remote_links[remote_idx] = 0.0
            return remote_idx

    def forward_connect(self):
        current_layer = self.parent.parent
        current_region = current_layer.parent
        current_layer_index = current_region.children.index(current_layer)

        if current_layer_index < len(current_region.children) - 1:
            next_layer_index = current_layer_index + 1
            next_layer = self.parent.parent.parent.children[next_layer_index]

            remote_column = random.choice(next_layer.children)
            remote_neuron = random.choice(remote_column.children)

            remote_idx = remote_neuron.try_connection(self.idx, 'forward')

            if remote_idx:
                logging.info("[CONNECTION|FORWARD] %s -> %s", self.idx, remote_idx)
                self.forward_links[remote_idx] = 0.0
                return remote_idx
