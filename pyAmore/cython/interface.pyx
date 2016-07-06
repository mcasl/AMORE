""" Amore: A module for training and simulating neural networks the way researchers need
"""
from common cimport RealNumber
import numpy as np
from .factories import AdaptiveGradientDescentMaterialsFactory
from .materials cimport MlpNetwork
from .network_fit_strategies cimport AdaptiveNetworkFitStrategy
from libc.stdio cimport printf

cpdef MlpNetwork mlp_network(list layers_size,
                             str hidden_layers_activation_function_name,
                             str output_layer_activation_function_name):
    factory = AdaptiveGradientDescentMaterialsFactory()
    builder = factory.make_network_builder()
    neural_network = builder.create_neural_network(layers_size,
                                                   hidden_layers_activation_function_name,
                                                   output_layer_activation_function_name)
    return neural_network

cpdef MlpNetwork fit_adaptive_gradient_descent(MlpNetwork mlp_neural_network,
                                               np.ndarray input_data,
                                               np.ndarray target_data,
                                               RealNumber learning_rate,
                                               int step_length,
                                               int number_of_steps):
    factory = AdaptiveGradientDescentMaterialsFactory()
    builder = factory.make_network_builder()
    builder.set_neurons_learning_rate(mlp_neural_network, learning_rate)
    cdef int dummy1, dummy2
    cdef AdaptiveNetworkFitStrategy fit_strategy = mlp_neural_network.fit_strategy

    for dummy1 in range(number_of_steps):
        for dummy2 in range(step_length):
            fit_strategy.fit(input_data, target_data)
        printf("Step=%d\n", dummy1)

    return mlp_neural_network


def network_weights(mlp_neural_network):
    def layer_weights(layer):
        weights = np.zeros((len(layer), len(layer[0].connections) + 1))

        for neuron_index, neuron in enumerate(layer):
            for connection_index, connection in enumerate(neuron.connections):
                weights[neuron_index, connection_index] = connection.weight
            weights[neuron_index, -1] = neuron.bias

        return weights

    return list(map(layer_weights, mlp_neural_network.layers[1:]))
