from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class Network(object, metaclass=ABCMeta):
    """ The mother of all neural networks (a.k.a Interface)
    """

    @abstractmethod
    def __init__(self, neural_factory):
        self.factory = neural_factory
        self.predict_strategy = neural_factory.make_neural_network_predict_strategy(self)
        self.fit_strategy = neural_factory.make_neural_network_fit_strategy(self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Method for obtaining outputs from inputs
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.__call__")

    @abstractmethod
    def read(self, input_data: np.ndarray):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.read")

    @abstractmethod
    def inspect_output(self) -> List[int]:
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.inspect_output")

    @property
    @abstractmethod
    def shape(self) -> List[int]:
        """ Gives information about the number of neurons in the neural network
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.shape")


class MlpNetwork(Network):
    """ Simple implementation of a multilayer feed forward network
    """

    def __init__(self, neural_factory):
        Network.__init__(self, neural_factory)
        self.layers = neural_factory.make_primitive_container()
        self.predict_strategy = neural_factory.make_neural_network_predict_strategy(self)

    # TODO:  cost_function = neural_factory.make_cost_function('LMS')

    def __call__(self, input_data: np.ndarray):
        return self.predict_strategy(input_data)

    def read(self, input_data: np.ndarray):
        for neuron, value in zip(self.layers[0], input_data):
            neuron.output = value

    def inspect_output(self):
        return [neuron.output for neuron in self.layers[-1]]

    @property
    def shape(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))


class Container(list):
    def __init__(self, iterable=[]):
        list.__init__(self, iterable)


class Neuron(object, metaclass=ABCMeta):
    """ The mother of all neurons (a.k.a. Interface)
    """

    @abstractmethod
    def __init__(self, neural_network: Network):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.output = 0.0
        self.neural_network = neural_network
        self.predict_strategy = neural_network.factory.make_neuron_predict_strategy(self)
        # self.fit_strategy should not be assigned here as it will depend on the neurons role
        # and it will be the builders's responsibility to assign it

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling Neuron.__call__()")


class MlpNeuron(Neuron):
    """ A simple neuron as in multilayer feed forward networks
    """

    def __init__(self, neural_network: MlpNetwork):
        """ Initializer. Python requires explicit call to base class initializer
        """
        Neuron.__init__(self, neural_network)
        self.activation_function = neural_network.factory.make_activation_function('default')
        self.connections = neural_network.factory.make_primitive_container()
        self.bias = 0.0

    def __call__(self, *args, **kwargs):
        return self.predict_strategy()


class Connection(object):
    """ A simple data structure for linking neurons
    """

    def __init__(self,
                 neuron: Neuron,
                 weight: float = 0.0):
        self.weight = weight
        self.neuron = neuron
