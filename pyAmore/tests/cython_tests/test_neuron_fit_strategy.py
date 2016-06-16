import unittest
import numpy as np
from pyAmore.cython.interface import *

from pyAmore.cython.neuron_fit_strategies import *


class TestNeuronFitStrategy(unittest.TestCase):
    def test_init(self):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        neuron = neural_network.layers[1][0]
        self.assertEqual(neuron.fit_strategy.neuron, neuron)
        self.assertEqual(neuron.fit_strategy.cost_function, cost_functions_set['default'])


class TestAdaptiveGradientDescentNeuronFitStrategy(unittest.TestCase):
    def test_init(self):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        neuron = neural_network.layers[1][0]
        self.assertEqual(neuron.fit_strategy.delta, 0.0)
        self.assertEqual(neuron.fit_strategy.learning_rate, 0.1)
        self.assertEqual(neuron.fit_strategy.output_derivative, 0.0)
        self.assertEqual(neuron.fit_strategy.target, 0.0)


class TestAdaptiveGradientDescentOutputNeuronFitStrategy(unittest.TestCase):
    def test_init(self):
        pass

    def test_call(self):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        neuron = neural_network.layers[2][0]
        input_data = np.random.rand(4, 3)
        neural_network(input_data)
        previous_weights_and_bias = [neuron.bias] + [connection.weight for connection in neuron.connections]
        neuron.fit_strategy.target = 0.6
        neuron.fit_strategy.fit()
        post_weights_and_bias = [neuron.bias] + [connection.weight for connection in neuron.connections]
        self.assertTrue(isinstance(neuron.fit_strategy, AdaptiveGradientDescentOutputNeuronFitStrategy))
        self.assertNotEqual(previous_weights_and_bias, post_weights_and_bias)


class TestAdaptiveGradientDescentHiddenNeuronFitStrategy(unittest.TestCase):
    def test_init__(self):
        pass

    def test_call(self, *args, **kwargs):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        neuron = neural_network.layers[1][0]
        input_data = np.random.rand(4, 3)
        neural_network(input_data)
        previous_weights_and_bias = [neuron.bias] + [connection.weight for connection in neuron.connections]
        neuron.fit_strategy.target = 0.6
        neuron.fit_strategy.delta = 0.2
        neuron.fit_strategy.fit()
        post_weights_and_bias = [neuron.bias] + [connection.weight for connection in neuron.connections]
        self.assertTrue(isinstance(neuron.fit_strategy, AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertNotEqual(previous_weights_and_bias, post_weights_and_bias)


if __name__ == '__main__':
    unittest.main()
