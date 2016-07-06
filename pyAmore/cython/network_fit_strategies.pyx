# cython: profile=True

from .neuron_predict_strategies import *
from .materials cimport MlpNeuron
from .neuron_fit_strategies cimport MlpNeuronFitStrategy

cdef class NetworkFitStrategy:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    cpdef fit(self, np.ndarray input_data, np.ndarray target_data):
        raise NotImplementedError("You shouldn't be calling NetworkFitStrategy.predict")


cdef class MlpNetworkFitStrategy(NetworkFitStrategy):
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)

    cpdef fit(self, np.ndarray input_data, np.ndarray target_data):
        raise NotImplementedError("You shouldn't be calling MlpNetworkFitStrategy.predict")

    cpdef poke_targets(self, data):
        output_layer = self.neural_network.layers[-1]
        cdef int neuron_position
        cdef int number_of_layers = len(output_layer)
        for neuron_position in range(number_of_layers):
            neuron = output_layer[neuron_position]
            value = data[neuron_position]
            neuron.fit_strategy.target = value

    cpdef backpropagate(self):
        cdef int layers_position
        cdef int neuron_position
        cdef int number_of_layers = len(self.neural_network.layers)
        cdef int number_of_neurons
        cdef MlpNeuron neuron
        cdef MlpNeuronFitStrategy neuron_fit_strategy

        for layers_position in range(-1 + number_of_layers, 0, -1):
            layer = self.neural_network.layers[layers_position]
            number_of_neurons = len(layer)
            for neuron_position in range(number_of_neurons):
                neuron = layer[neuron_position]
                neuron_fit_strategy = neuron.fit_strategy
                neuron_fit_strategy.fit()

                #for layer in reversed(self.neural_network.layers):
                #    for neuron in layer:
                #        neuron.fit_strategy.fit()



cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    cpdef fit(self, np.ndarray input_data, np.ndarray target_data):
        cdef int position
        cdef int input_data_length = len(input_data)
        for position in range(input_data_length):
            input_data_row = input_data[position]
            target_data_row = target_data[position]
            self.neural_network.poke_inputs(input_data_row)
            self.neural_network.predict_strategy.activate_neurons()
            self.poke_targets(target_data_row)
            self.backpropagate()
