from abc import ABCMeta, abstractmethod


class NeuralNetworkBuilder(object, metaclass=ABCMeta):
    """ The mother of all neural creators (a.k.a. Interface)
    """

    @abstractmethod
    def create_neural_network(self, *args):
        raise NotImplementedError("You shouldn't be calling NeuralNetworkBuilder.create_neural_network")
