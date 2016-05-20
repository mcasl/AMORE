from abc import abstractmethod

from amore.network_elements.container import Container
from amore.network_elements.connection import Connection
from amore.network_elements.mlp_neuron import MlpNeuron
from amore.network_elements.mlp_neural_network import MlpNeuralNetwork
from amore.network_elements.activation_functions import activation_functions_set
from amore.network_elements.cost_functions import cost_functions_set
from amore.builders.mlp_builder import MlpNeuralNetworkBuilder

from .neural_factory import NeuralFactory


class MlpFactory(NeuralFactory):
    """ Simple implementation of a factories of multilayer feed forward network's elements
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("You shouldn't be calling MlpFactory.__init__")

    @staticmethod
    def make_primitive_connection(neuron):
        return Connection(neuron)

    @staticmethod
    def make_primitive_container():
        return Container()

    @staticmethod
    def make_primitive_neuron(neural_network):
        neuron = MlpNeuron(neural_network)
        return neuron

    def make_primitive_layer(self, size, neural_network):
        layer = self.make_primitive_container()
        for dummy in range(size):
            layer.append(self.make_primitive_neuron(neural_network))
        return layer

    def make_primitive_neural_network(self):
        neural_network = MlpNeuralNetwork(self)
        neural_network.predict_strategy = self.make_neural_network_predict_strategy(neural_network)
        neural_network.layers = self.make_primitive_container()
        return neural_network

    @staticmethod
    def make_neural_network_builder():
        return MlpNeuralNetworkBuilder()

    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neuron_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neuron_fit_strategy")

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neural_network):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neural_network_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neural_network_fit_strategy")

    @staticmethod
    def make_activation_function(function_name):
        return activation_functions_set[function_name]

    @staticmethod
    def make_cost_function(function_name):
        return cost_functions_set[function_name]
