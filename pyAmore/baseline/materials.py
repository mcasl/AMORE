from abc import ABCMeta, abstractmethod
import numpy as np
from .activation_functions import activation_functions_set
from .cost_functions import cost_functions_set


class Network(metaclass=ABCMeta):
    """ The mother of all neural networks (a.k.a Interface)
    """

    @abstractmethod
    def __init__(self, neural_factory):
        self.factory = neural_factory
        self.predict_strategy = neural_factory.make_network_predict_strategy(self)
        self.fit_strategy = neural_factory.make_network_fit_strategy(self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Method for obtaining outputs from inputs
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.predict")

    @abstractmethod
    def poke_inputs(self, input_data):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.poke_inputs")

    @abstractmethod
    def pick_outputs(self):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.pick_outputs")

    @property
    @abstractmethod
    def shape(self):
        """ Gives information about the number of neurons in the neural network
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.shape")


class MlpNetwork(Network):
    """ Simple implementation of a multilayer feed forward network
    """

    def __init__(self, neural_factory):
        Network.__init__(self, neural_factory)
        self.layers = neural_factory.make_primitive_container()

    def __call__(self, input_data):
        return self.predict_strategy(input_data)

    def poke_inputs(self, input_data):
        for neuron, value in zip(self.layers[0], input_data):
            neuron.output = value

    def pick_outputs(self):
        return [neuron.output for neuron in self.layers[-1]]

    @property
    def shape(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))


class MlpContainer(list):
    def __init__(self, iterable=None):
        if iterable is None:
            iterable = []
        # noinspection PyTypeChecker
        list.__init__(self, iterable)


class Neuron(metaclass=ABCMeta):
    """ The mother of all neurons (a.k.a. Interface)

    """

    @abstractmethod
    def __init__(self, neural_network):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.output = 0.0
        self.neural_network = neural_network
        self.predict_strategy = None
        self.fit_strategy = None
        # self.fit_strategy should not be assigned here as it might depend on the neurons role
        # and it will be the builders's responsibility to assign it
        # Similarly, self.predict_strategy is not assigned here for versatility.
        # It's the builder that assigns it.

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling Neuron.predict()")


class MlpNeuron(Neuron):
    """ A simple neuron as in multilayer feed forward networks
    """

    def __init__(self, neural_network):
        """ Initializer. Python requires explicit call to base class initializer
        """
        Neuron.__init__(self, neural_network)
        self.activation_function = neural_network.factory.make_activation_function('default')
        self.connections = neural_network.factory.make_primitive_container()
        self.bias = 0.0

    def __call__(self, *args, **kwargs):
        return self.predict_strategy.predict()


class MlpConnection:
    """ A simple data structure for linking neurons in MLP networks
    """

    def __init__(self, neuron, weight=0.0):
        self.weight = weight
        self.neuron = neuron
