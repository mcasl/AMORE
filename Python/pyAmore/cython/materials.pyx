# cython: profile=True

from common                     cimport RealNumber
from network_fit_strategies     cimport NetworkFitStrategy
from network_predict_strategies cimport NetworkPredictStrategy
from neuron_fit_strategies      cimport NeuronFitStrategy
from neuron_predict_strategies  cimport NeuronPredictStrategy
from activation_functions       cimport ActivationFunction
cimport numpy as np

cdef class MlpContainer(list):
    pass

cdef class Network:
    """ The mother of all neural networks (a.k.a Interface)
    """

    def __init__(self, neural_factory):
        self.factory = neural_factory
        self.predict_strategy = neural_factory.make_network_predict_strategy(self)
        self.fit_strategy = neural_factory.make_network_fit_strategy(self)

    def __call__(self, *args, **kwargs):
        """ Method for obtaining outputs from inputs
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.predict")

    cpdef poke_inputs(self, np.ndarray input_data):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.poke_inputs")

    cpdef pick_outputs(self):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.pick_outputs")

    @property
    def shape(self):
        """ Gives information about the number of neurons in the neural network
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.shape")

cdef class MlpNetwork(Network):
    """ Simple implementation of a multilayer feed forward network
    """

    def __init__(self, neural_factory):
        Network.__init__(self, neural_factory)
        self.layers = neural_factory.make_primitive_container()

    def __call__(self, np.ndarray input_data):
        return self.predict_strategy(input_data)

    cpdef poke_inputs(self, np.ndarray input_data):
        cdef int neuron_position
        cdef number_of_neurons = len(self.layers[0])
        cdef MlpNeuron neuron
        cdef MlpContainer input_layer = self.layers[0]
        for neuron_position in range(number_of_neurons):
            neuron = input_layer[neuron_position]
            value = input_data[neuron_position]
            neuron.output = value

    cpdef pick_outputs(self):
        return [neuron.output for neuron in self.layers[-1]]

    @property
    def shape(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))

cdef class Neuron:
    """ The mother of all neurons (a.k.a. Interface)
    """

    def __init__(self, Network neural_network):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.output = 0.0

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling Neuron.predict()")

cdef class MlpNeuron(Neuron):
    """ A simple neuron as in multilayer feed forward networks
    """

    def __init__(self, MlpNetwork neural_network):
        """ Initializer. Python requires explicit call to base class initializer
        """
        Neuron.__init__(self, neural_network)
        self.neural_network = neural_network
        self.predict_strategy = None
        self.fit_strategy = None
        # self.fit_strategy should not be assigned here as it might depend on the neurons role
        # and it will be the builders's responsibility to assign it
        # Similarly, self.predict_strategy is not assigned here for versatility.
        # It's the builder that assigns it.

        self.activation_function = neural_network.factory.make_activation_function('default')
        self.connections = neural_network.factory.make_primitive_container()
        self.bias = 0.0

    def __call__(self, *args, **kwargs):
        return self.predict_strategy.predict()

cdef class MlpConnection:
    """ A simple data structure for linking neurons in MLP networks
    """

    def __init__(self, neuron, weight=0.0):
        self.weight = weight
        self.neuron = neuron
