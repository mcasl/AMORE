import unittest

from amore.interface import *
from amore.materials import *
from amore.network_fit_strategies import *
from amore.neuron_fit_strategies import *


class TestNetworkFitStrategies(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        fit_strategy = factory.make_neural_network_fit_strategy(neural_network)
        self.assertEqual(fit_strategy.neural_network, neural_network)

    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        fit_strategy = factory.make_neural_network_fit_strategy(neural_network)
        self.assertRaises(NotImplementedError, NetworkFitStrategy.__call__, fit_strategy)


class TestMlpNetworkFitStrategy(unittest.TestCase):
    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        fit_strategy = factory.make_neural_network_fit_strategy(neural_network)
        self.assertRaises(NotImplementedError, MlpNetworkFitStrategy.__call__, fit_strategy)

    def test_write_targets_in_output_layer(self):
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        data = [1, 2, 3, 4, 5]
        neural_network.fit_strategy.write_targets_in_output_layer(data)
        result = [neuron.fit_strategy.target for neuron in neural_network.layers[-1]]
        self.assertEqual(data, result)

    def test_single_pattern_backward_action(self):
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        previous = []
        for layer in neural_network.layers:
            for neuron in layer:
                for connection in neuron.connections:
                    previous.append(connection.weight)
                previous.append(neuron.bias)

        target_data = [1, 2, 3, 4, 5]
        neural_network.fit_strategy.write_targets_in_output_layer(target_data)
        neural_network.fit_strategy.single_pattern_backward_action()
        post = []
        for layer in neural_network.layers:
            for neuron in layer:
                for connection in neuron.connections:
                    post.append(connection.weight)
                post.append(neuron.bias)
        self.assertNotEqual(post, previous)

    def test_set_neurons_fit_strategy(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        neural_network.fit_strategy.set_neurons_fit_strategy(factory)
        result = []
        for layer in neural_network.layers:
            for neuron in layer:
                result.append(neuron.fit_strategy)
        self.assertTrue(isinstance(result[0], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[1], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[2], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[3], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[4], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[5], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[6], AdaptiveGradientDescentHiddenNeuronFitStrategy))
        self.assertTrue(isinstance(result[7], AdaptiveGradientDescentOutputNeuronFitStrategy))
        self.assertTrue(isinstance(result[8], AdaptiveGradientDescentOutputNeuronFitStrategy))
        self.assertTrue(isinstance(result[9], AdaptiveGradientDescentOutputNeuronFitStrategy))
        self.assertTrue(isinstance(result[10], AdaptiveGradientDescentOutputNeuronFitStrategy))
        self.assertTrue(isinstance(result[11], AdaptiveGradientDescentOutputNeuronFitStrategy))

    def test_set_neurons_learning_rate(self):
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        neural_network.fit_strategy.set_neurons_learning_rate(1234)
        rates = []
        for layer in neural_network.layers:
            for neuron in layer:
                rates.append(neuron.fit_strategy.learning_rate)
        self.assertEqual(rates, [1234, 1234, 1234,
                                 1234, 1234, 1234, 1234,
                                 1234, 1234, 1234, 1234, 1234])


class TestAdaptiveGradientDescentNetworkFitStrategy(unittest.TestCase):
    def test_call(self):
        input_data = np.random.rand(100, 1)
        target_data = input_data ** 2
        neural_network = mlp_network([1, 4, 1], 'tanh', 'identity')
        neural_network.fit_strategy(input_data, target_data)
        self.assertEqual(1, 1)  # TODO: change test
