""" Amore: A module for training and simulating neural networks the way researchers need
"""

from activation_functions import *
from container import *
from cost_functions import *
from factories import *
from network_elements import *
from network_fit_strategies import *
from network_predict_strategies import *
from neural_network_builders import *
from neuron_fit_strategy import *
from neuron_predict_strategies import *


# TODO: remember: alternative constructors @classmethod def myalternativeconstructor(class, other arguments):


def mlp_network(layers_size,
                hidden_layers_activation_function_name,
                output_layer_activation_function_name):
    factory = AdaptiveGradientDescentFactory()
    builder = factory.make_neural_network_builder()
    net = builder.create_neural_network(factory, layers_size, hidden_layers_activation_function_name,
                                        output_layer_activation_function_name)
    return net
