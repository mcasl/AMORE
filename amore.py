import random
from abc import ABCMeta, abstractmethod
from math import sqrt

from activation_functions import *
from container import Container


class Connection:
    """ A simple data structure for linking neurons."""
    def __init__(self, neuron, weight=0.0):
        """
        Initializer.
        :param neuron: Input neuron
        :param weight: float value
        :return: A Connection instance
        """
        self.weight = weight
        self.neuron = neuron

    def __repr__(self):
        """ Pretty print """
        return '\nFrom:\t {label} \t Weight= \t {weight}'.format(label=self.neuron.label, weight=self.weight)


class Neuron(metaclass=ABCMeta):
    """ The mother of all neurons (a.k.a. Interface)"""

    @abstractmethod
    def __init__(self):
        """
        Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.induced_local_field = 0.0
        self.output = 0.0
        self.target = 0.0
        self.connections = Container()

    @abstractmethod
    def __repr__(self):
        """ Pretty print """
        pass


class SimpleNeuron(Neuron):
    """ A simple neuron as in multilayer feed forward networks"""

    def __init__(self):
        """
        Initializer. Python requires explicit call to base class initializer.
        """
        Neuron.__init__(self)

    def __repr__(self):
        """ Pretty print """

        result = ('\n\n'
                  '-----------------------------------\n'
                  ' Id label: {label}\n'
                  '-----------------------------------\n'
                  ' Output: {output}\n'
                  '-----------------------------------\n'
                  # TODO:                  '{predict_behavior}'
                  ' Target: {target}\n'
                  '-----------------------------------\n').format(label=self.label,
                                                                  output=self.output,
                                                                  # TODO:                                                                predict_behavior=repr(self.predictBehavior),
                                                                  target=self.target)
        result += repr(self.connections)
        #          '\n-----------------------------------\n'
        #                   'Neuron Train Behavior: {train_behavior}'.format(train_behavior=self.train_behavior),
        #         '\n-----------------------------------'
        return result


class NeuralNetwork(metaclass=ABCMeta):
    """ The mother of all neural networks (a.k.a Interface) """

    @abstractmethod
    def train(self, *args):
        """
        Method for training the network
        """
        pass

    @abstractmethod
    def sim(self, *args):
        """
        Method for obtaining outputs from inputs
        """
        pass

    @abstractmethod
    def number_of_neurons(self):
        """
        Gives information about the number of neurons in the neural network
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass


class SimpleNeuralNetwork(NeuralNetwork):
    """ Simple implementation of a multilayer feed forward network """

    def __init__(self, neural_factory):
        """ Initializer
        """
        self.layers = neural_factory.make_container()

    # TODO:  cost_function = neural_factory.make_cost_function('LMS')

    def train(self, *args):
        pass

    def sim(self, *args):
        pass

    def number_of_neurons(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))

    def __repr__(self):
        """ Pretty print """
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


"""

def sim(numericMatrix) {

    bool checkIncorrectNumberOfRows(
            inputSize() != static_cast<size_type>(numericMatrix.nrow()))
    if (checkIncorrectNumberOfRows) {
        throw std::runtime_error(
                "\nIncorrect number or rows. The number of input neurons must be equal to the number of rows of the input matrix.\n")
    }

    Rcpp::NumericMatrix outputMatrix(outputSize(), numericMatrix.ncol())
    std::vector<double>::iterator inputIterator(numericMatrix.begin())
    std::vector<double>::iterator outputIterator(outputMatrix.begin())

    // PREDICT LOOP
    {
        NeuronIterator inputNeuronIterator(d_inputLayer.createIterator())
        NeuronIterator outputNeuronIterator(d_outputLayer.createIterator())
        for (int i = 0 i < numericMatrix.ncol() i++) {
            writeInput(inputIterator, inputNeuronIterator)
            singlePatternForwardAction()
            readOutput(outputIterator, outputNeuronIterator)
        }
        delete outputNeuronIterator
        delete inputNeuronIterator
    }
    return outputMatrix

}



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
            NeuronTrainBehavior neuronTrainBehavior(neuralFactory.makeOutputNeuronTrainBehavior(neuronIterator.currentItem()))
            neuronIterator.currentItem().setNeuronTrainBehavior(neuronTrainBehavior)
        }
        delete neuronIterator
    }
}


def writeInput(std::vector<double>::iterator& iterator, NeuronIterator neuronIterator) {
    for (neuronIterator.first() !neuronIterator.isDone()
            neuronIterator.next()) {
        neuronIterator.currentItem().setOutput(*iterator++)
    }
}


def writeTarget(iterator, neuron_container):
    for neuron_iter in neuron_container:
        neuron.setTarget(*iterator++)


void SimpleNetwork::singlePatternForwardAction() {
    // Hidden Layers
    {
        LayerIterator layerIterator(d_hiddenLayers.createIterator())
        for (layerIterator.first() !layerIterator.isDone()	layerIterator.next()) {
            NeuronIterator neuronIterator(	layerIterator.currentItem().createIterator())
            for (neuronIterator.first() !neuronIterator.isDone()	neuronIterator.next()) {
                neuronIterator.currentItem().singlePatternForwardAction()
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
            neuronIterator.currentItem().singlePatternForwardAction()
        }
        delete neuronIterator
    }
}

void SimpleNetwork::singlePatternBackwardAction() {
    // Output Layers
    {
        NeuronIterator neuronIterator(
                d_outputLayer.createReverseIterator())
        for (neuronIterator.first() !neuronIterator.isDone()
                neuronIterator.next()) {
            neuronIterator.currentItem().singlePatternBackwardAction()
        }
        delete neuronIterator
    }
    // Hidden Layers
    {
        LayerIterator layerIterator(
                d_hiddenLayers.createReverseIterator())
        for (layerIterator.first() !layerIterator.isDone()
                layerIterator.next()) {
            NeuronIterator neuronIterator(
                    layerIterator.currentItem().createReverseIterator())
            for (neuronIterator.first() !neuronIterator.isDone()
                    neuronIterator.next()) {
                neuronIterator.currentItem().singlePatternBackwardAction()
            }
            delete neuronIterator
        }
        delete layerIterator
    }
}

void SimpleNetwork::readOutput(std::vector<double>::iterator& iterator, NeuronIterator neuronIterator) {
    for (neuronIterator.first() !neuronIterator.isDone()
            neuronIterator.next()) {
        *iterator++ = neuronIterator.currentItem().d_output
    }
}

Rcpp::List SimpleNetwork::train(Rcpp::List parameterList) {
    return d_networkTrainBehavior.train(parameterList)

}

size_type SimpleNetwork::inputSize() {
    return d_inputLayer.size()
}

size_type SimpleNetwork::outputSize() {
    return d_outputLayer.size()
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

void SimpleNetwork::show() {
    Rprintf("\n\n=========================================================\n")
    Rprintf("         Neural Network")
    Rprintf("\n=========================================================")

    Rprintf("\n Input size: %d\n", inputSize())
    Rprintf("\n Output size: %d\n", outputSize())
    Rprintf("\n Network Train Behavior: %s\n",
            getNetworkTrainBehaviorName().c_str()) // TODO revisar si esto es un memory-leak
    Rprintf("\n Cost Function: %s\n", getCostFunctionName().c_str()) // TODO revisar si esto es un memory-leak

bool SimpleNetwork::validate() {
    d_inputLayer.validate()
    d_hiddenLayers.validate()
    d_outputLayer.validate()
    return true
}


"""


class NeuralFactory(metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)"""

    @abstractmethod
    def make_connection(self, neuron):
        pass

    @abstractmethod
    def make_container(self):
        pass

    @abstractmethod
    def make_neuron(self, label, neuron_container, total_amount_of_parameters, neural_network):
        pass

    # TODO:    @abstractmethod
    # TODO:    def make_predict_behavior(self, neuron):
    # TODO:        pass

    @abstractmethod
    def make_neural_network(self, neural_factory):
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
    """ Simple implementation of a factory of multilayer feed forward network's elements    """

    def make_connection(self, neuron: Neuron) -> Connection:
        return Connection(neuron)

    def make_container(self) -> Container:
        return Container()

    def make_neuron(self, label, neuron_container=None, total_amount_of_parameters=None, neural_network=None):
        neuron = SimpleNeuron()
        neuron.label = label
        # TODO:       _neuron.predict_behavior = self.make_predict_behavior(_neuron)
        # TODO:       _neuron.activation_function = self.make_activation_function(_neuron,f0, f1)
        # TODO:       _neuron.train_behavior = self.make_hidden_neuron_train_behavior()

        if (neuron_container is None) or (total_amount_of_parameters is None) or (neural_network is None):
            return neuron
        else:
            neuron.neural_network = neural_network
            extreme = sqrt(3.0 / total_amount_of_parameters)
            for neuron_iter in neuron_container:
                connection = self.make_connection(neuron_iter)
                connection.weight = random.uniform(-extreme, extreme)
                neuron.connections.append(connection)
            # TODO:            _neuron.predict_behavior.bias = random.uniform(-extreme, extreme)
            return neuron

            # TODO:    def make_predict_behavior(self, neuron):
            # TODO:        return MLP_behavior(neuron)

    def make_neural_network(self, neural_factory):
        simple_neural_network = SimpleNeuralNetwork(neural_factory)
        # TODO: simple_neural_network.network_train_behavior = self.make_network_train_behavior(simple_neural_network)
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


class NeuralCreator(metaclass=ABCMeta):
    """ The mother of all neural creators (a.k.a. Interface)"""
    @abstractmethod
    def create_neural_network(self, *args):
        pass


class SimpleNeuralCreator(NeuralCreator):
    """ A simple implementation of the logic for building multilayer feed forward networks """
    def create_neural_network(self, neural_factory, number_of_neurons,
                              hidden_layers_activation_function_name,
                              output_layer_activation_function_name):
        """ A method for creating a multilayer feed forward network
        :param neural_factory:  A factory such as MlpFactory
        :param number_of_neurons: A list of integers describing the number of neurons in each layer
        :param hidden_layers_activation_function_name: Function name according to those comprised in activation_functions.py
        :param output_layer_activation_function_name: Function name according to those comprised in activation_functions.py
        :return: A multilayer feed forward neural network
        """
        neural_network = neural_factory.make_neural_network()
        if len(number_of_neurons) < 2:
            raise ValueError('[create_feed_forward_network]: Error, number of layers lower than 2.')
        SimpleNeuralCreator.populate_network(neural_network, number_of_neurons)
        SimpleNeuralCreator.fully_connect_network(neural_factory, neural_network)
        SimpleNeuralCreator.initialize_weights_and_biases(neural_network)
        return neural_network

    @staticmethod
    def populate_network(neural_factory, neural_network, number_of_neurons):
        """ This method fills the neural network with neurons according to
            the structure given in the number_of_neurons list.
            The neurons are unconnected yet and their weights are uninitialized.
        :param neural_factory:  A factory such as MlpFactory
        :param neural_network: A multilayer feed forward network
        :param number_of_neurons: A list of integers describing the number of neurons in each layer
        """
        neuron_label = 0
        layers = neural_factory.make_container()
        for layer_size in number_of_neurons:
            layer = neural_factory.make_container()
            for dummy_iter in range(layer_size):
                neuron = neural_factory.make_neuron(neuron_label)
                neuron.neural_network = neural_network
                layer.append(neuron)
                neuron_label += 1
            layers.append(layer)
        neural_network.layers = layers

    @staticmethod
    def connect_two_layers(neural_factory, first, second):
        """
        This subroutine links two layers of neurons in a
        fully connected manner
        :param neural_factory:  A factory such as MlpFactory
        :param first: Layer on the left
        :param second: Layer on the right
        """
        for destination in second:
            destination.connections = neural_factory.make_container()
            for origin in first:
                destination.connections.append(neural_factory.make_connection(origin))

    @staticmethod
    def fully_connect_network(neural_factory, neural_network):
        """ This subroutine walks the neurons through
            and establishes the connections in a fully connected manner
            :param neural_factory:  A factory such as MlpFactory
            :param neural_network: A multilayer feed forward network
        """
        first = neural_network.layers[0]
        for second in neural_network.layers[1:]:
            SimpleNeuralCreator.connect_two_layers(neural_factory, first, second)
            first = second

    @staticmethod
    def initialize_weights_and_biases(neural_network):
        """ This subroutine walks the neurons through
            and changes the connections' weights following a recipe
            given in Simon Haykin's book so as to improve the learning phase
            :param neural_network: A multilayer feed forward network
        """
        number_of_neurons = neural_network.number_of_neurons()
        # Calculation of the total amount of parameters
        total_number_of_neurons = sum(number_of_neurons)
        total_amount_of_parameters = 0
        previous_number = 0
        for current_number in number_of_neurons:
            total_amount_of_parameters += current_number * previous_number
            previous_number = current_number
        total_amount_of_parameters += total_number_of_neurons
        extreme = sqrt(3.0 / total_amount_of_parameters)

        for layer in neural_network.layers:
            for neuron in layer:
                for connection in neuron.connections:
                    connection.weight = random.uniform(-extreme, extreme)
                neuron.bias = random.uniform(-extreme, extreme)
