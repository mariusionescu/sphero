class BaseSensor(object):
    def __init__(self, name):
        self.name = name


class Vision(BaseSensor):
    def send(self, data):
        pass


class Sound(BaseSensor):
    def send(self, data):
        pass


class Text(BaseSensor):
    def send(self, data):
        pass
