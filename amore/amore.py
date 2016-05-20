""" Amore: A module for training and simulating neural networks the way researchers need
"""

from cost_functions import *
from network_elements import *
from neural_network_builders import *
from neuron_predict_strategies import *

from amore.factory.adapt_gd_factory import *


# TODO: remember: alternative constructors @classmethod def myalternativeconstructor(class, other arguments):


def mlp_network(layers_size,
                hidden_layers_activation_function_name,
                output_layer_activation_function_name):
    factory = AdaptiveGradientDescentFactory()
    builder = factory.make_neural_network_builder()
    net = builder.create_neural_network(factory,
                                        layers_size,
                                        hidden_layers_activation_function_name,
                                        output_layer_activation_function_name)
    return net


def fit_adaptive_gradient_descent(mlp_neural_network,
                                  input_data,
                                  target_data,
                                  learning_rate,
                                  step_length,
                                  number_of_steps):
    mlp_neural_network.fit_strategy.set_neurons_learning_rate(learning_rate)
    for step in range(number_of_steps):
        for inner_iterations in range(step_length):
            mlp_neural_network.fit_strategy(input_data, target_data)
        print("Step={step}".format(step=step))
    return mlp_neural_network


if __name__ == '__main__':
    import numpy as np

    input_data = np.random.rand(1000, 1)
    target = input_data ** 2
    net = mlp_network([1, 5, 1],
                      'tanh',
                      'identity')
    fit_adaptive_gradient_descent(net,
                                  input_data,
                                  target,
                                  0.1,
                                  100,
                                  30)
