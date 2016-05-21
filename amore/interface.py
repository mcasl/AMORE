""" Amore: A module for training and simulating neural networks the way researchers need
"""
from .factories import *
from .neuron_fit_strategies import *

# TODO: remember: alternative constructors @classmethod def myalternativeconstructor(class, other arguments):


def mlp_network(layers_size,
                hidden_layers_activation_function_name,
                output_layer_activation_function_name):
    factory = AdaptiveGradientDescentFactory()
    builder = factory.make_neural_network_builder()
    neural_network = builder.create_neural_network(factory,
                                                   layers_size,
                                                   hidden_layers_activation_function_name,
                                                   output_layer_activation_function_name)
    return neural_network


def fit_adaptive_gradient_descent(mlp_neural_network: MlpNeuralNetwork,
                                  input_data: np.array,
                                  target_data: np.array,
                                  learning_rate: float,
                                  step_length: int,
                                  number_of_steps: int) -> MlpNeuralNetwork:
    mlp_neural_network.fit_strategy.set_neurons_learning_rate(learning_rate)
    for step in range(number_of_steps):
        for inner_iterations in range(step_length):
            mlp_neural_network.fit_strategy(input_data, target_data)
        print("Step={step}".format(step=step))
    return mlp_neural_network


if __name__ == '__main__':
    import numpy as np

    data = np.random.rand(1000, 1)
    target = data ** 2
    net = mlp_network([1, 5, 1],
                      'tanh',
                      'identity')
    fit_adaptive_gradient_descent(net,
                                  data,
                                  target,
                                  0.1,
                                  100,
                                  30)
