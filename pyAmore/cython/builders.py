import math
import random
from typing import List

from abc import ABCMeta, abstractmethod
from .materials import MlpNetwork


class NetworkBuilder(object, metaclass=ABCMeta):
    """ The mother of all neural creators (a.k.a. Interface)
    """

    @abstractmethod
    def create_neural_network(self, *args):
        raise NotImplementedError("You shouldn't be calling NeuralNetworkBuilder.create_neural_network")


class MlpNetworkBuilder(NetworkBuilder):
    """ A simple implementation of the logic for building multilayer feed forward networks
    """

    def create_neural_network(self,
                              neural_factory,
                              layers_size: List[int],
                              hidden_layers_activation_function_name: str,
                              output_layer_activation_function_name: str):

        """ A method for creating a multilayer feed forward network
        """
        neural_network = neural_factory.make_primitive_network()
        if layers_size:
            MlpNetworkBuilder.create_primitive_layers(neural_factory, neural_network, layers_size)
            MlpNetworkBuilder.create_neuron_fit_and_predict_sequence(neural_network)
            MlpNetworkBuilder.connect_network_layers(neural_factory, neural_network)
            MlpNetworkBuilder.initialize_network(neural_network)
            MlpNetworkBuilder.set_neurons_predict_strategy(neural_factory, neural_network)
            MlpNetworkBuilder.set_neurons_fit_strategy(neural_factory, neural_network)
            return neural_network

    @staticmethod
    def set_neurons_fit_strategy(neural_factory,
                                 neural_network: MlpNetwork):
        # Providing input neurons with a fit strategy simplifies
        # the code for adjusting the weights and biases
        for layer in neural_network.layers:
            for neuron in layer:
                neuron.fit_strategy = neural_factory.make_neuron_fit_strategy(neuron)

    @staticmethod
    def set_neurons_predict_strategy(neural_factory,
                                     neural_network: MlpNetwork):
        # Providing input neurons with a fit strategy simplifies
        # the code for adjusting the weights and biases
        for layer in neural_network.layers:
            for neuron in layer:
                neuron.predict_strategy = neural_factory.make_neuron_predict_strategy(neuron)

    @staticmethod
    def set_neurons_learning_rate(neural_network, learning_rate):
        for neuron in neural_network.fit_strategy.neuron_fit_sequence:
            neuron.fit_strategy.learning_rate = learning_rate

    @staticmethod
    def create_primitive_layers(neural_factory,
                                neural_network,
                                layers_size: List[int]):
        """ This method fills the neural network with neurons according to
            the structure given in the number_of_neurons list.
            The neurons are unconnected yet and their weights are uninitialized.
        :param neural_factory:  A factories such as MlpFactory
        :param neural_network: A multilayer feed forward network
        :param layers_size: A list of integers describing each layer shape
        """
        layers = neural_factory.make_primitive_container()
        for size in layers_size:
            layer = neural_factory.make_primitive_container()
            for dummy in range(size):
                layer.append(neural_factory.make_primitive_neuron(neural_network))
            layers.append(layer)
        neural_network.layers = layers

    @staticmethod
    def connect_network_layers(neural_factory,
                               neural_network):
        """ This subroutine walks the neurons through
            and establishes the connections in a fully connected manner.
            :param neural_factory:  A factories such as MlpFactory
            :param neural_network: A multilayer feed forward network
        """
        origin_layer = neural_network.layers[0]
        for destination_layer in neural_network.layers[1:]:
            for neuron in destination_layer:
                neuron.connections = neural_factory.make_primitive_container()
                for origin in origin_layer:
                    neuron.connections.append(neural_factory.make_primitive_connection(origin))
            origin_layer = destination_layer

    @staticmethod
    def create_neuron_fit_and_predict_sequence(neural_network):
        predict_sequence = neural_network.factory.make_primitive_container()
        for layer in neural_network.layers[1:]:
            for neuron in layer:
                predict_sequence.append(neuron)
        neural_network.predict_strategy.neuron_predict_sequence = predict_sequence
        neural_network.fit_strategy.neuron_fit_sequence = neural_network.factory.make_primitive_container()
        neural_network.fit_strategy.neuron_fit_sequence.extend(reversed(predict_sequence))

    @staticmethod
    def initialize_network(neural_network):
        """ This subroutine walks the neurons through
            and changes:
                *   The connections' weights following a recipe
                    given in Simon Haykin's book so as to improve the learning phase
                *   The neuron labels
                *   The neuron.neural_network attribute
            :param neural_network: A multilayer feed forward network
        """
        neural_network_shape_is_not_valid = (neural_network.shape == []) or (neural_network.shape == [0])
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
