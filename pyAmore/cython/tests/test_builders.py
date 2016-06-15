import unittest
from pyAmore.cython.interface import *
from pyAmore.cython.builders import MlpNetworkBuilder


class TestBuilders(unittest.TestCase):
    def test_set_neurons_learning_rate(self):
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        MlpNetworkBuilder.set_neurons_learning_rate(neural_network, 1234)
        rates = []
        for layer in neural_network.layers[1:]:
            for neuron in layer:
                rates.append(neuron.fit_strategy.learning_rate)
        self.assertEqual(rates, [1234, 1234, 1234, 1234,
                                 1234, 1234, 1234, 1234, 1234])
