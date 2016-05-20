from abc import ABCMeta, abstractmethod


class NetworkFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkFitStrategy.__call__")
