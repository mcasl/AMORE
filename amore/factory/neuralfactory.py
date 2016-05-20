from abc import ABCMeta, abstractmethod


class NeuralFactory(object, metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.__init__")

    @staticmethod
    @abstractmethod
    def make_neural_network_builder():
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_builder")

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_fit_strategy")

    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_fit_strategy")
