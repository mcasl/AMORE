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

    def __repr__(self):
        """ Pretty print. Probably to be moved to a view class (mvc)
        """
        return '\nFrom:\t {label} \t Weight= \t {weight}'.format(label=self.neuron.label, weight=self.weight)


class Neuron(object, metaclass=ABCMeta):
    """ The mother of all neurons (a.k.a. Interface)
    """

    @abstractmethod
    def __init__(self):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.connections = Container()

    @abstractmethod
    def __repr__(self):
        """ Pretty print. Probably to be moved to a view class (mvc)
        """
        pass

    @abstractmethod
    def propagate(self):
        """ Neuron forward update by applying its activation function
        """
        pass


class SimpleNeuron(Neuron):
    """ A simple neuron as in multilayer feed forward networks
    """

    def __init__(self):
        """ Initializer. Python requires explicit call to base class initializer
        """
        Neuron.__init__(self)
        self.induced_local_field = 0.0
        self.output = 0.0
        self.target = 0.0
        self.bias = 0.0
        self.activation_function = activation_function_set['default']
        self.activation_function_derivative = activation_function_derivative_set['default']

    def __repr__(self):
        """ Pretty print. Probably to be moved to a view class (mvc)
        """

        result = ('\n\n'
                  '-----------------------------------\n'
                  ' Id label: {label}\n'
                  '-----------------------------------\n'
                  ' Output: {output}\n'
                  '-----------------------------------\n'
                  # TODO:    '{predict_behavior}'
                  ' Target: {target}\n'
                  '-----------------------------------\n').format(label=self.label,
                                                                  output=self.output,
                                                                  # TODO:  predict_behavior=repr(self.predictBehavior),
                                                                  target=self.target)
        result += repr(self.connections)
        #          '\n-----------------------------------\n'
        #                   'Neuron Train Behavior: {train_behavior}'.format(train_behavior=self.train_behavior),
        #         '\n-----------------------------------'
        return result

    def propagate(self):
        """ Neuron forward update by applying its activation function
        """
        inputs_x_weights = map((lambda connection: connection.neuron.output * connection.weight), self.connections)
        self.induced_local_field = reduce(operator.add, inputs_x_weights)
        self.induced_local_field += self.bias
        self.output = self.activation_function()


class NeuralNetwork(object, metaclass=ABCMeta):
    """ The mother of all neural networks (a.k.a Interface)
    """

    @abstractmethod
    def train(self, *args):
        """ Method for training the network
        """
        pass

    @abstractmethod
    def sim(self, *args):
        """ Method for obtaining outputs from inputs
        """
        pass

    @abstractmethod
    def insert_input_data(self, *args):
        pass

    @property
    @abstractmethod
    def shape(self):
        """ Gives information about the number of neurons in the neural network
        """
        pass

    @abstractmethod
    def __repr__(self):
        """ Pretty print. Probably to be moved to a view class (mvc)
        """
        pass


class SimpleNeuralNetwork(NeuralNetwork):
    """ Simple implementation of a multilayer feed forward network
    """

    def __init__(self, neural_factory):
        """ Initializer
        """
        self.layers = neural_factory.make_container()

    # TODO:  cost_function = neural_factory.make_cost_function('LMS')

    def train(self, *args):
        pass

    def sim(self, input_data):
        data_number_of_rows, data_number_of_cols = input_data.shape
        input_layer_size, *hidden_layers_size, output_layer_size = self.shape

        if input_layer_size != data_number_of_cols:
            raise ValueError(
                '\n[SimpleNeuralNetwork.sim Error:] Input layer size different from data number of columns\n')

        output_data = numpy.zeros(data_number_of_rows, output_layer_size)
        for row, input_data, in enumerate(input_data):
            self.insert_input_data(input_data)
            self.single_pattern_forward_action()
            output_data[row, :] = self.read_output_layer()
        return output_data

    @property
    def shape(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))

    def __repr__(self):
        """ Pretty print
        """
        result = ('\n----------------------------------------------\n'
                  'Simple Neural Network\n'
                  '----------------------------------------------\n'
                  '     INPUT LAYER:\n'
                  '----------------------------------------------\n'
                  )
        result += repr(self.layers[0])
        result += ('\n----------------------------------------------\n'
                   '     HIDDEN LAYERS:\n'
                   '----------------------------------------------\n'
                   )
        result += repr(self.layers[1:-1])
        result += ('\n----------------------------------------------\n'
                   '     OUTPUT LAYER:\n'
                   '----------------------------------------------\n'
                   )
        result += repr(self.layers[-1])
        return result

    def insert_input_data(self, data):
        for neuron, value in zip(self.layers[0], data):
            neuron.output = value

    def write_targets_in_output_layer(self, data):
        for neuron, value in zip(self.layers[-1], data):
            neuron.target = value

    def single_pattern_forward_action(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.propagate()

    def single_pattern_backward_action(self):
        for layer in reversed(self.layers):
            for neuron in layer:
                neuron.single_pattern_backward_action()

    def read_output_layer(self):
        return [neuron.output for neuron in self.layers[-1]]


"""


def setNeuronTrainBehavior(neural_factory) {
    // Hidden Layers
    {
        LayerIterator layerIterator(d_hiddenLayers.createIterator())
        for (layerIterator.first() !layerIterator.isDone()
                layerIterator.next()) {
            NeuronIterator neuronIterator(
                    layerIterator.currentItem().createIterator())
            for (neuronIterator.first() !neuronIterator.isDone()
                    neuronIterator.next()) {
                NeuronTrainBehavior neuronTrainBehavior(
                        neuralFactory.makeHiddenNeuronTrainBehavior(
                                neuronIterator.currentItem()))
                neuronIterator.currentItem().setNeuronTrainBehavior(
                        neuronTrainBehavior)
            }
            delete neuronIterator
        }
        delete layerIterator
    }
    // Output Layers
    {
        NeuronIterator neuronIterator(d_outputLayer.createIterator())
        for (neuronIterator.first() !neuronIterator.isDone() neuronIterator.next()) {
            neuronTrainBehavior(neuralFactory.makeOutputNeuronTrainBehavior(neuronIterator.currentItem())
            neuronIterator.currentItem().setNeuronTrainBehavior(neuronTrainBehavior)
        }
        delete neuronIterator
    }
}

    Rcpp::List SimpleNetwork::train(Rcpp::List parameterList) {
        return d_networkTrainBehavior.train(parameterList)

    }



double SimpleNetwork::costFunctionf0(double output, double target) {
    return d_costFunction.f0(output, target)
}

double SimpleNetwork::costFunctionf1(double output, double target) {
    return d_costFunction.f1(output, target)
}

void SimpleNetwork::setLearningRate(double learningRate) {

    // Hidden Layers
    {
        LayerIterator layerIterator(d_hiddenLayers.createIterator())
        for (layerIterator.first() !layerIterator.isDone()
                layerIterator.next()) {
            NeuronIterator neuronIterator(
                    layerIterator.currentItem().createIterator())
            for (neuronIterator.first() !neuronIterator.isDone()
                    neuronIterator.next()) {
                neuronIterator.currentItem().setLearningRate(learningRate)
            }
            delete neuronIterator
        }
        delete layerIterator
    }
    // Output Layers
    {
        NeuronIterator neuronIterator(d_outputLayer.createIterator())
        for (neuronIterator.first() !neuronIterator.isDone()
                neuronIterator.next()) {
            neuronIterator.currentItem().setLearningRate(learningRate)
        }
        delete neuronIterator
    }
}


"""


class NeuralFactory(object, metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)
    """

    @abstractmethod
    def make_connection(self, neuron):
        pass

    @abstractmethod
    def make_container(self):
        pass

    @abstractmethod
    def make_primitive_neuron(self, label, neuron_container, total_amount_of_parameters, neural_network):
        pass

    @abstractmethod
    def make_primitive_layer(self, size):
        pass

    # TODO:    @abstractmethod
    # TODO:    def make_predict_behavior(self, neuron):
    # TODO:        pass

    @abstractmethod
    def make_primitive_neural_network(self, neural_factory):
        pass

    @abstractmethod
    def make_neural_creator(self):
        pass

    @abstractmethod
    def make_activation_function(self, function_name):
        pass

    @abstractmethod
    def make_network_train_behavior(self, neural_network):
        pass

    @abstractmethod
    def make_output_neuron_train_behavior(self, neuron):
        pass

    @abstractmethod
    def make_hidden_neuron_train_behavior(self, neuron):
        pass

    @abstractmethod
    def make_cost_function(self, function_name):
        pass


class MlpFactory(NeuralFactory):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

    def make_connection(self, neuron: Neuron) -> Connection:
        return Connection(neuron)

    def make_container(self):
        return Container()

    def make_primitive_neuron(self):
        neuron = SimpleNeuron()
        return neuron

    def make_primitive_layer(self, size):
        layer = self.make_container()
        for neuron in range(size):
            layer.append(self.make_primitive_neuron())
        return layer

    def make_primitive_neural_network(self):
        simple_neural_network = SimpleNeuralNetwork(self)
        return simple_neural_network

    def make_neural_creator(self):
        return SimpleNeuralCreator()

    def make_activation_function(self, function_name):
        return activation_function_set[function_name]

    def make_network_train_behavior(self, neural_network):
        pass

    def make_output_neuron_train_behavior(self, neuron):
        pass

    def make_hidden_neuron_train_behavior(self, neuron):
        pass

    def make_cost_function(self, function_name):
        pass


"""
MLPfactory::makeCostFunction(std::string functionName)
  'LMS'
  'LMLS'
   'TAO'
   else
       throw std::invalid_argument(
           "[SimpleNetwork::train Error]: Unknown cost function.")
"""


class NeuralCreator(object, metaclass=ABCMeta):
    """ The mother of all neural creators (a.k.a. Interface)
    """

    @abstractmethod
    def create_neural_network(self, *args):
        pass


class SimpleNeuralCreator(NeuralCreator):
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
            SimpleNeuralCreator.create_primitive_layers(neural_factory, primitive_neural_network, layers_size)
            SimpleNeuralCreator.connect_network_layers(neural_factory, primitive_neural_network)
            SimpleNeuralCreator.initialize_network(primitive_neural_network)
        return primitive_neural_network

    @staticmethod
    def create_primitive_layers(neural_factory: NeuralFactory,
                                neural_network: NeuralNetwork,
                                layers_size: list):
        """ This method fills the neural network with neurons according to
            the structure given in the number_of_neurons list.
            The neurons are unconnected yet and their weights are uninitialized.
        :param neural_factory:  A factory such as MlpFactory
        :param neural_network: A multilayer feed forward network
        :param layers_size: A list of integers describing each layer shape
        """
        layers = neural_factory.make_container()
        for size in layers_size:
            layer = neural_factory.make_primitive_layer(size)
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
                neuron.connections = neural_factory.make_container()
                for origin in origin_layer:
                    neuron.connections.append(neural_factory.make_connection(origin))
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
