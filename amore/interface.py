""" Amore: A module for training and simulating neural networks the way researchers need
"""
from .factories import *
from .neuron_fit_strategies import *
from .materials import MlpNeuralNetwork
import matplotlib.pyplot as plt

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


def fit_adaptive_gradient_descent(mlp_neural_network: object,
                                  input_data: object,
                                  target_data: object,
                                  learning_rate: object,
                                  step_length: object,
                                  number_of_steps: object) -> object:
    mlp_neural_network.fit_strategy.set_neurons_learning_rate(learning_rate)
    for step in range(number_of_steps):
        for inner_iterations in range(step_length):
            mlp_neural_network.fit_strategy(input_data, target_data)
        plt.scatter(input_data, mlp_neural_network(input_data))
        print("Step={step}".format(step=step))
    return mlp_neural_network

