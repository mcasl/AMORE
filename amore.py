""" Amore: A module for training and simulating neural networks the way researchers need


"""

import operator
import random
from abc import ABCMeta, abstractmethod
from functools import reduce
from math import sqrt

import numpy

from activation_functions import *
from container import Container
from cost_functions import *


# TODO: remember: alternative constructors @classmethod def myalternativeconstructor(class, other arguments):

class Connection(object):
    """ A simple data structure for linking neurons
    """

    def __init__(self, neuron, weight=0.0):
        """
        Initializer.
        :param neuron: Input neuron
        :param weight: float value
        :return: An initialized Connection instance
        """
        self.weight = weight
        self.neuron = neuron


class Neuron(object, metaclass=ABCMeta):
    """ The mother of all neurons (a.k.a. Interface)
    """

    @abstractmethod
    def __init__(self, neural_network):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.neural_network = neural_network
        self.predict_strategy = neural_network.factory.make_neuron_predict_strategy(self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MlpNeuron(Neuron):
    """ A simple neuron as in multilayer feed forward networks
    """

    def __init__(self, neural_network):
        """ Initializer. Python requires explicit call to base class initializer
        """
        Neuron.__init__(self, neural_network)
        self.induced_local_field = 0.0
        self.output = 0.0
        self.target = 0.0
        self.activation_function = neural_network.factory.make_activation_function('default')
        self.activation_function_derivative = neural_network.factory.make_activation_function_derivative('default')
        self.connections = neural_network.factory.make_primitive_container()

    def __call__(self, *args, **kwargs):
        return self.predict_strategy()


class NeuralNetwork(object, metaclass=ABCMeta):
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
        pass

    @abstractmethod
    def read_input_data(self, data):
        pass

    @abstractmethod
    def inspect_output(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        """ Gives information about the number of neurons in the neural network
        """
        pass


class MlpNeuralNetwork(NeuralNetwork):
    """ Simple implementation of a multilayer feed forward network
    """

    def __init__(self, neural_factory):
        NeuralNetwork.__init__(self, neural_factory)
        self.layers = neural_factory.make_primitive_container()

    # TODO:  cost_function = neural_factory.make_cost_function('LMS')

    def __call__(self, input_data):
        return self.predict_strategy()

    def read_input_data(self, data):
        for neuron, value in zip(self.layers[0], data):
            neuron.output = value

    def inspect_output(self):
        return [neuron.output for neuron in self.layers[-1]]

    @property
    def shape(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))


class NeuralFactory(object, metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)
    """

    @staticmethod
    @abstractmethod
    def make_neural_network_builder():
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        pass


    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        pass


class MlpFactory(NeuralFactory):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

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
        neural_network.fit_strategy = self.make_neural_network_fit_strategy(neural_network)
        neural_network.layers = self.make_primitive_container()
        return neural_network

    @staticmethod
    def make_neural_network_builder():
        return MlpNeuralNetworkBuilder()

    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neural_network):
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        pass

    @staticmethod
    def make_activation_function(function_name):
        return activation_functions_set[function_name]

    @staticmethod
    def make_activation_function_derivative(function_name):
        return activation_functions_derivative_set[function_name]

    @staticmethod
    def make_cost_function(function_name):
        return cost_functions_set[function_name]

    @staticmethod
    def make_cost_function_derivative(function_name):
        return cost_functions_derivative_set[function_name]


class AdaptiveGradientDescentFactory(MlpFactory):
    @staticmethod
    def make_neuron_predict_strategy(neuron):
        return AdaptiveGradientDescentNeuronPredictStrategy(neuron)

    @staticmethod
    def make_neuron_fit_strategy(neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return AdaptiveGradientDescentOutputNeuronFitStrategy(neuron)
        else:
            return AdaptiveGradientDescentHiddenNeuronFitStrategy(neuron)

    @staticmethod
    def make_neural_network_predict_strategy(neural_network):
        return AdaptiveGradientDescentNetworkPredictStrategy(neural_network)

    @staticmethod
    def make_neural_network_fit_strategy(neural_network):
        return AdaptiveGradientDescentNetworkFitStrategy(neural_network)


#
# class AdaptiveGradientDescentWithMomentumFactory(MlpFactory):
#
#     def make_neuron_fit_strategy(self, neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return AdaptiveGradientDescentWithMomentumOutputNeuronFitStrategy(neuron)
#         else:
#             return AdaptiveGradientDescentWithMomentumHiddenNeuronFitStrategy(neuron)
#
#     def make_neural_network_predict_strategy(self, neural_network):
#         return AdaptiveGradientDescentWithMomentumNetworkPredictStrategy(neural_network)
#
#
# class BatchGradientDescentFactory(MlpFactory):
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return BatchGradientDescentOutputNeuronFitStrategy(neuron)
#         else:
#             return BatchGradientDescentHiddenNeuronFitStrategy(neuron)
#
#
#     def make_neural_network_predict_strategy(self, neural_network):
#         return BatchGradientDescentNetworkPredictStrategy(neural_network)
#
#
# class BatchGradientDescentWithMomentumFactory(MlpFactory):
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return BatchDescentWithMomentumOutputNeuronFitStrategy(neuron)
#         else:
#             return BatchDescentWithMomentumHiddenNeuronFitStrategy(neuron)
#
#
#     def make_neural_network_predict_strategy(self, neural_network):
#         return BatchDescentWithMomentumNetworkPredictStrategy(neural_network)


class NeuralNetworkBuilder(object, metaclass=ABCMeta):
    """ The mother of all neural creators (a.k.a. Interface)
    """

    @abstractmethod
    def create_neural_network(self, *args):
        pass


class MlpNeuralNetworkBuilder(NeuralNetworkBuilder):
    """ A simple implementation of the logic for building multilayer feed forward networks
    """

    def create_neural_network(self,
                              neural_factory,
                              layers_size,
                              hidden_layers_activation_function_name,
                              output_layer_activation_function_name) -> NeuralNetwork:

        """ A method for creating a multilayer feed forward network
        :param neural_factory:  A factory such as MlpFactory
        :param layers_size: A list of integers describing the number of neurons in each layer
        :param hidden_layers_activation_function_name: According to activation_functions.py
        :param output_layer_activation_function_name: According to activation_functions.py
        :return: A multilayer feed forward neural network
        """
        primitive_neural_network = neural_factory.make_primitive_neural_network()
        if layers_size:
            MlpNeuralNetworkBuilder.create_primitive_layers(neural_factory, primitive_neural_network, layers_size)
            MlpNeuralNetworkBuilder.connect_network_layers(neural_factory, primitive_neural_network)
            MlpNeuralNetworkBuilder.initialize_network(primitive_neural_network)
        return primitive_neural_network

    @staticmethod
    def create_primitive_layers(neural_factory,
                                neural_network,
                                layers_size):
        """ This method fills the neural network with neurons according to
            the structure given in the number_of_neurons list.
            The neurons are unconnected yet and their weights are uninitialized.
        :param neural_factory:  A factory such as MlpFactory
        :param neural_network: A multilayer feed forward network
        :param layers_size: A list of integers describing each layer shape
        """
        layers = neural_factory.make_primitive_container()
        for size in layers_size:
            layer = neural_factory.make_primitive_layer(size, neural_network)
            layers.append(layer)
        neural_network.layers = layers

    @staticmethod
    def connect_network_layers(neural_factory, neural_network):
        """ This subroutine walks the neurons through
            and establishes the connections in a fully connected manner.
            :param neural_factory:  A factory such as MlpFactory
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
    def initialize_network(neural_network):
        """ This subroutine walks the neurons through
            and changes:
                *   The connections' weights following a recipe
                    given in Simon Haykin's book so as to improve the learning phase
                *   The neuron labels
                *   The neuron.neural_network attribute
            :param neural_network: A multilayer feed forward network
        """

        if neural_network.shape == [0]:
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
        extreme = sqrt(3.0 / total_amount_of_parameters)

        # Network walk through
        label_number = 0
        for layer in neural_network.layers:
            for neuron in layer:
                # Change weights and bias
                for connection in neuron.connections:
                    connection.weight = random.uniform(-extreme, extreme)
                neuron.bias = random.uniform(-extreme, extreme)

                # Initialize other neuron attributes
                neuron.label, label_number = label_number, label_number + 1
                neuron.neural_network = neural_network
                # TODO : more to come, remember to change the comments above


class NeuronPredictStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neuron):
        self.neuron = neuron

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    def __init__(self, neuron):
        NeuronPredictStrategy.__init__(self, neuron)
        self.bias = 0.0

    def __call__(self, *args, **kwargs):
        inputs_x_weights = map((lambda connection: connection.neuron.output * connection.weight),
                               self.neuron.connections)
        self.neuron.induced_local_field = reduce(operator.add, inputs_x_weights) + self.bias
        self.neuron.output = self.neuron.activation_function(self.neuron.induced_local_field)
        self.neuron.output_derivative = self.neuron.activation_function_derivative(self.neuron.induced_local_field)
        return self.neuron.output


class AdaptiveGradientDescentNeuronPredictStrategy(MlpNeuronPredictStrategy):
    def __init__(self, neuron):
        MlpNeuronPredictStrategy.__init__(self, neuron)


class AdaptiveGradientDescentWithMomentumNeuronPredictStrategy(MlpNeuronPredictStrategy):
    def __init__(self, neuron):
        MlpNeuronPredictStrategy.__init__(self, neuron)


class BatchGradientDescentNeuronPredictStrategy(MlpNeuronPredictStrategy):
    def __init__(self, neuron):
        MlpNeuronPredictStrategy.__init__(self, neuron)


class BatchGradientDescentWithMomentumNeuronPredictStrategy(MlpNeuronPredictStrategy):
    def __init__(self, neuron):
        MlpNeuronPredictStrategy.__init__(self, neuron)


class NetworkPredictStrategy(object, metaclass=ABCMeta):
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def __call__(self, *args, **kwargs):
        pass


class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    def __init__(self, neural_network):
        NetworkPredictStrategy.__init__(self, neural_network)

    def __call__(self, input_data):
        data_number_of_rows, data_number_of_cols = input_data.shape
        input_layer_size, *hidden_layers_size, output_layer_size = self.neural_network.shape

        if input_layer_size != data_number_of_cols:
            raise ValueError(
                '\n[SimpleNeuralNetwork.sim Error:] Input layer size different from data number of columns\n')

        output_data = numpy.zeros((data_number_of_rows, output_layer_size))
        for row, input_data in enumerate(input_data):
            self.neural_network.read_input_data(input_data)
            self.neural_network.activate_neurons()
            output_data[row, :] = self.neural_network.inspect_output_layer()
        return output_data


class AdaptiveGradientDescentNetworkPredictStrategy(MlpNetworkPredictStrategy):
    def __init__(self, neural_network):
        MlpNetworkPredictStrategy.__init__(self, neural_network)


class AdaptiveGradientDescentWithMomentumNetworkPredictStrategy(MlpNetworkPredictStrategy):
    def __init__(self, neural_network):
        MlpNetworkPredictStrategy.__init__(self, neural_network)


class BatchGradientDescentNetworkPredictStrategy(MlpNetworkPredictStrategy):
    def __init__(self, neural_network):
        MlpNetworkPredictStrategy.__init__(self, neural_network)


class BatchGradientDescentWithMomentumNetworkPredictStrategy(MlpNetworkPredictStrategy):
    def __init__(self, neural_network):
        MlpNetworkPredictStrategy.__init__(self, neural_network)

class NeuralViewer(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def show_connection(connection):
        pass

    @staticmethod
    @abstractmethod
    def show_neuron(neuron):
        pass

    @staticmethod
    @abstractmethod
    def show_neural_network(neural_network):
        pass


class MlpNeuralViewer(NeuralViewer):
    @staticmethod
    def show_connection(connection):
        print('\nFrom:\t {label} \t Weight= \t {weight}'.format(label=connection.neuron.label,
                                                                weight=connection.weight))

    @staticmethod
    def show_neuron(neuron):
        result = ('\n\n'
                  '-----------------------------------\n'
                  ' Id label: {label}\n'
                  '-----------------------------------\n'
                  ' Output: {output}\n'
                  '-----------------------------------\n'
                  # TODO:    '{predict_behavior}'
                  ' Target: {target}\n'
                  '-----------------------------------\n').format(label=neuron.label,
                                                                  output=neuron.output,
                                                                  # TODO:  predict_behavior=repr(self.predictBehavior),
                                                                  target=neuron.target)
        result += repr(neuron.connections)
        #          '\n-----------------------------------\n'
        #                   'Neuron Train Behavior: {train_behavior}'.format(train_behavior=self.train_behavior),
        #         '\n-----------------------------------'
        print(result)

    @staticmethod
    def show_neural_network(neural_network):
        """ Pretty print
        """
        result = ('\n----------------------------------------------\n'
                  'Simple Neural Network\n'
                  '----------------------------------------------\n'
                  '     INPUT LAYER:\n'
                  '----------------------------------------------\n'
                  )
        result += repr(neural_network.layers[0])
        result += ('\n----------------------------------------------\n'
                   '     HIDDEN LAYERS:\n'
                   '----------------------------------------------\n'
                   )
        result += repr(neural_network.layers[1:-1])
        result += ('\n----------------------------------------------\n'
                   '     OUTPUT LAYER:\n'
                   '----------------------------------------------\n'
                   )
        result += repr(neural_network.layers[-1])
        print(result)


class NeuronFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neuron):
        neuron = neuron

    def cost_function(self, output, target):
        return self.cost_function(self.neuron.output, self.neuron.target)

    def cost_function_derivative(self, output, target):
        return self.cost_function_derivative(self.neuron.output, self.neuron.target);


class MlpNeuronFitStrategy(NeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        NeuronFitStrategy.__init__(self, neuron)


class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        MlpNeuronFitStrategy.__init__(neuron)
        self.delta = 0.0
        self.learning_rate = 0.0


class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(neuron)

    def __call__(self):
        self.delta = self.neuron.output_derivative * self.neuron.cost_function_derivative(self.neuron.output,
                                                                                          self.neuron.target)
        minus_learning_rate_x_delta = -self.learning_rate * self.delta
        self.bias += minus_learning_rate_x_delta
        for connection in self.neuron.connections:
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0


class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(neuron)

    def __call__(self, *args, **kwargs):
        self.delta = self.neuron.output_derivative * self.delta
        minus_learning_rate_x_delta = - self.learning_rate * self.delta
        self.bias += minus_learning_rate_x_delta
        for connection in self.neuron.connections:
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0


class NetworkFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MlpNetworkFitStrategy(NetworkFitStrategy):
    @abstractmethod
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def write_targets_in_output_layer(self, data):
        for neuron, value in zip(self.neural_network.layers[-1], data):
            neuron.target = value

    def activate_neurons(self):
        for layer in self.neural_network.layers[1:]:
            for neuron in layer:
                neuron()

    def single_pattern_backward_action(self):
        for layer in reversed(self.neural_network.layers):
            for neuron in layer:
                neuron.fit()

    def set_neurons_fit_strategy(self, neural_factory):
        # Hidden Layers
        for layer in self.neural_network.layers[1:-1]:
            for neuron in layer:
                neuron.fit_strategy = neural_factory.make_hidden_neuron_fit_strategy(neuron)
        # Output Layers
        for neuron in self.neural_network.layers[-1]:
            neuron.fit_strategy = neural_factory.make_output_neuron_fit_strategy(neuron)

    def set_neurons_learning_rate(self, learning_rate):
        for layer in self.neural_network.layers[1:]:
            for neuron in layer:
                neuron.learning_rate = learning_rate

    def set_learning_rate(self, learning_rate):
        self.neural_network.set_learning_rate(learning_rate)


class AdaptiveGradientDescentNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    def __call__(self, *args, **kwargs):
        self.set_learning_rate(learning_rate)
        max_shows = number_of_epochs / show_step if number_of_epochs > show_step else 1

        for dummy_1 in range(max_shows):
            for dummy_2 in range(show_step):
                for input_data_row, target_data_row in zip(input_data, target_data):
                    self.neural_network.read_input_data(input_data_row)
                    self.neural_network.activate_neurons()
                    self.neural_network.write_target(target_data_row)
                    self.neural_network.single_pattern_backward_action()


def mlp_network(layers_size,
                hidden_layers_activation_function_name,
                output_layer_activation_function_name):
    factory = MlpFactory()
    builder = factory.make_neural_network_builder()
    net = builder.create_neural_network(factory, layers_size, hidden_layers_activation_function_name,
                                        output_layer_activation_function_name)
    return net
