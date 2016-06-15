""" Amore: A module for training and simulating neural networks the way researchers need
"""
import numpy as np
from pyAmore.cython.factories import AdaptiveGradientDescentMaterialsFactory
from pyAmore.cython.materials import MlpNetwork
from pyAmore.cython.builders import MlpNetworkBuilder

# TODO: remember: alternative constructors @classmethod def myalternativeconstructor(class, other arguments):


def mlp_network(layers_size,
                hidden_layers_activation_function_name,
                output_layer_activation_function_name):
    factory = AdaptiveGradientDescentMaterialsFactory()
    builder = factory.make_network_builder()
    neural_network = builder.create_neural_network(factory,
                                                   layers_size,
                                                   hidden_layers_activation_function_name,
                                                   output_layer_activation_function_name)
    return neural_network


def fit_adaptive_gradient_descent(mlp_neural_network,
                                  input_data,
                                  target_data,
                                  learning_rate,
                                  step_length,
                                  number_of_steps) -> MlpNetwork:
    MlpNetworkBuilder.set_neurons_learning_rate(mlp_neural_network, learning_rate)
    for step in range(number_of_steps):
        for inner_iterations in range(step_length):
            mlp_neural_network.fit_strategy(input_data, target_data)
        print("Step={step}".format(step=step))
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
