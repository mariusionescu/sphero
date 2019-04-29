from base import BaseElement
import logging
import random


class Neuron(BaseElement):

    LEARNING_RATE = 0.01
    FORGET_RATE = 0.1
    CHARGE_LOSS_RATIO = 0.2
    CHARGE_ACTIVATION_THRESHOLD = 0.5
    FORWARD_ACTIVATION_THRESHOLD = 0.4

    def __init__(self, cortex, parent):
        super(Neuron, self).__init__(cortex, parent)
        self.charge = 0.0
        self.set_idx()

    def backward_optimization(self, idx, learning_rate):
        strength = self.links[idx]
        if learning_rate < 0:
            new_strength = max(0.0, strength + learning_rate)
            self.links[idx] = new_strength
            logging.debug("[BACKWARD|FORGET] adjusted link %s -> %s (%s)",
                          self.idx, idx, self.links[idx])
        else:
            new_strength = min(0.99, strength + learning_rate)
            self.links[idx] = new_strength
            logging.info("[BACKWARD|LEARNING] adjusted link %s -> %s (%s)",
                         self.idx, idx, self.links[idx])

    async def fire(self, strength=0.0, remote_idx=None, neuronal=True):

        # if self.charge < self.ACTIVATION_THRESHOLD:
        self.charge = min(self.charge + strength * self.CHARGE_LOSS_RATIO, 0.99)

        logging.info("[NEURON|CHARGE] %s -- new charge %s", self.idx, self.charge)

        charge_fire = False

        if self.charge > self.CHARGE_ACTIVATION_THRESHOLD:
            charge_fire = True

        for neuron_idx, forward_strength in self.links.items():

            if forward_strength > self.FORWARD_ACTIVATION_THRESHOLD:
                self.parent.parent.activated_neurons.add(self.idx)

                neuron = self.get_neuron(neuron_idx)

                new_strength = max(0.0, forward_strength - self.FORGET_RATE)

                future = neuron.fire(strength=new_strength, remote_idx=self.idx)
                logging.info("[FUTURE] queueing %s -> %s (%s)", self.idx, neuron_idx, new_strength)
                self.cortex.queue.put(future)

            elif self.charge > self.CHARGE_ACTIVATION_THRESHOLD:
                neuron = self.get_neuron(neuron_idx)

                new_charge = max(0.0, strength * (1 - self.CHARGE_LOSS_RATIO))

                future = neuron.fire(strength=new_charge, remote_idx=self.idx)
                logging.info("[FUTURE] queueing %s -> %s (%s)", self.idx, neuron_idx, new_charge)
                self.cortex.queue.put(future)

        if charge_fire:
            self.charge = 0.0

        if not charge_fire and neuronal and remote_idx:
            for neuron_idx in self.back_links:
                neuron = self.get_neuron(neuron_idx)
                if neuron.idx == remote_idx:
                    neuron.backward_optimization(self.idx, self.LEARNING_RATE)
                else:
                    neuron.backward_optimization(self.idx, -self.FORGET_RATE)

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

    def try_connection(self, remote_idx):
        if (
                remote_idx != self.idx and
                remote_idx not in self.links and
                remote_idx not in self.back_links
        ):
            self.back_links.append(remote_idx)
            return self.idx
        else:
            logging.info("[NEURON %s] already has a connection to %s", self.idx, remote_idx)

    def local_connect(self):
        remote_neuron = random.choice(self.parent.children)
        remote_idx = remote_neuron.try_connection(self.idx)

        if remote_idx:
            logging.info("[CONNECTION|LOCAL] %s -> %s", self.idx, remote_idx)
            self.links[remote_idx] = 0.0
            return remote_idx

    def remote_connect(self):
        remote_column = random.choice(self.parent.parent.children)
        remote_neuron = random.choice(remote_column.children)

        remote_idx = remote_neuron.try_connection(self.idx)

        if remote_idx:
            logging.info("[CONNECTION|REMOTE] %s -> %s", self.idx, remote_idx)
            self.links[remote_idx] = 0.0
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

            remote_idx = remote_neuron.try_connection(self.idx)

            if remote_idx:
                logging.info("[CONNECTION|FORWARD] %s -> %s", self.idx, remote_idx)
                self.links[remote_idx] = 0.0
                return remote_idx
