import math
import random
from typing import List

from abc import ABCMeta, abstractmethod
from .materials import MlpNetwork


class NetworkBuilder(metaclass=ABCMeta):
    """ The mother of all neural creators (a.k.a. Interface)
    """

    @abstractmethod
    def __init__(self, factory):
        self.factory = factory

    @abstractmethod
    def create_neural_network(self, *args):
        raise NotImplementedError("You shouldn't be calling NeuralNetworkBuilder.create_neural_network")


class MlpNetworkBuilder(NetworkBuilder):
    """ A simple implementation of the logic for building multilayer feed forward networks
    """

    def __init__(self, factory):
        NetworkBuilder.__init__(self, factory)

    def create_neural_network(self,
                              layers_size,
                              hidden_layers_activation_function_name,
                              output_layer_activation_function_name):

        """ A method for creating a multilayer feed forward network
        """
        neural_network = self.factory.make_primitive_network()

        if layers_size:
            self.create_primitive_layers(neural_network, layers_size)
            self.connect_network_layers(neural_network)
            self.initialize_network(neural_network)
            self.set_neurons_predict_strategy(neural_network)
            self.set_neurons_fit_strategy(neural_network)
            return neural_network

    def set_neurons_fit_strategy(self, neural_network):
        # Providing input neurons with a fit strategy simplifies
        # the code for adjusting the weights and biases
        for layer in neural_network.layers:
            for neuron in layer:
                neuron.fit_strategy = self.factory.make_neuron_fit_strategy(neuron)

    def set_neurons_predict_strategy(self, neural_network):
        # Providing input neurons with a fit strategy simplifies
        # the code for adjusting the weights and biases
        for layer in neural_network.layers:
            for neuron in layer:
                neuron.predict_strategy = self.factory.make_neuron_predict_strategy(neuron)

    def set_neurons_learning_rate(self, neural_network, learning_rate):
        for layer in neural_network.layers:
            for neuron in layer:
                neuron.fit_strategy.learning_rate = learning_rate

    def create_primitive_layers(self, neural_network, layers_size):
        """ This method fills the neural network with neurons according to
            the structure given in the number_of_neurons list.
            The neurons are unconnected yet and their weights are uninitialized.
        :param neural_network: A multilayer feed forward network
        :param layers_size: A list of integers describing each layer shape
        """
        layers = self.factory.make_primitive_container()
        for size in layers_size:
            layer = self.factory.make_primitive_container()
            for _ in range(size):
                layer.append(self.factory.make_primitive_neuron(neural_network))
            layers.append(layer)
        neural_network.layers = layers

    def connect_network_layers(self, neural_network):
        """ This subroutine walks the neurons through
            and establishes the connections in a fully connected manner.
            :param neural_factory:  A factories such as MlpFactory
            :param neural_network: A multilayer feed forward network
        """
        origin_layer = neural_network.layers[0]
        for destination_layer in neural_network.layers[1:]:
            for neuron in destination_layer:
                neuron.connections = self.factory.make_primitive_container()
                for origin in origin_layer:
                    neuron.connections.append(self.factory.make_primitive_connection(origin))
            origin_layer = destination_layer

    def initialize_network(self, neural_network):
        """ This subroutine walks the neurons through
            and changes:
                *   The connections' weights following a recipe
                    given in Simon Haykin's book so as to improve the learning phase
                *   The neuron labels
                *   The neuron.neural_network attribute
            :param neural_network: A multilayer feed forward network
        """
        neural_network_shape_is_not_valid = neural_network.shape in [[], [0]]
        if neural_network_shape_is_not_valid:
            raise ValueError('[Initialize network]: Empty net,  Shape is [0].')
        # Calculation of the total amount of parameters

        number_of_neurons = neural_network.shape
        total_number_of_neurons = sum(number_of_neurons)
        total_amount_of_parameters = 0
        previous_number = 0
        for current_number in number_of_neurons:
            total_amount_of_parameters += current_number * previous_number
            previous_number = current_number
        total_amount_of_parameters += total_number_of_neurons
        extreme = math.sqrt(3.0 / total_amount_of_parameters)

        # Network walk through
        label_number = 0
        for layer in neural_network.layers:
            for neuron in layer:
                # Change weights and bias
                for connection in neuron.connections:
                    connection.weight = random.uniform(-extreme, extreme)
                neuron.bias = random.uniform(-extreme, extreme)

                # Initialize other neuron attributes
                neuron.label, label_number = str(label_number), label_number + 1
                neuron.neural_network = neural_network
