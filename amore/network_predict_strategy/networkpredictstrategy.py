from abc import ABCMeta, abstractmethod


class NetworkPredictStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkPredictStrategy.__call__")

    @abstractmethod
    def activate_neurons(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkPredictStrategy.activate_neurons")
