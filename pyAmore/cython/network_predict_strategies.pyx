# cython: profile=True

import numpy as np

from .neuron_predict_strategies import *
from .materials cimport MlpNeuron, MlpContainer

cdef class NetworkPredictStrategy:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkPredictStrategy.predict")

    cpdef activate_neurons(self):
        raise NotImplementedError("You shouldn't be calling NetworkPredictStrategy.activate_neurons")


cdef class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    def __init__(self, neural_network):
        NetworkPredictStrategy.__init__(self, neural_network)

    def __call__(self, input_data):
        data_number_of_rows, data_number_of_columns = input_data.shape
        input_layer_size, *hidden_layers_size, output_layer_size = self.neural_network.shape

        if input_layer_size != data_number_of_columns:
            raise ValueError(
                '\n[SimpleNeuralNetwork.sim Error:] Input layer size different from data number of columns\n')

        output_data = np.zeros((data_number_of_rows, output_layer_size))
        cdef int position
        cdef int number_of_rows = data_number_of_rows
        for position in range(number_of_rows):
            row_data = input_data[position]
            self.neural_network.poke_inputs(row_data)
            self.activate_neurons()
            output_data[position, :] = self.neural_network.pick_outputs()
        return output_data

    cpdef activate_neurons(self):
        cdef int layer_position
        cdef int neuron_position
        cdef int number_of_layers = len(self.neural_network.layers)
        cdef int number_of_neurons
        cdef MlpNeuron neuron
        cdef MlpContainer layer
        for layer_position in range(1, number_of_layers):
            layer = self.neural_network.layers[layer_position]
            number_of_neurons = len(layer)
            for neuron_position in range(number_of_neurons):
                neuron = layer[neuron_position]
                neuron.predict_strategy.predict()
