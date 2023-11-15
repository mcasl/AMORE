import unittest
from pyAmore.baseline.interface import *


class TestBuilders(unittest.TestCase):
    def test_set_neurons_learning_rate(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        builder = factory.make_network_builder()
        builder.set_neurons_learning_rate(neural_network, 1234)
        rates = []
        for layer in neural_network.layers[1:]:
            rates.extend(neuron.fit_strategy.learning_rate for neuron in layer)
        self.assertEqual(rates, [1234, 1234, 1234, 1234,
                                 1234, 1234, 1234, 1234, 1234])
