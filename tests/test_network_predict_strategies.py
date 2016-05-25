import unittest

from amore.interface import *
from amore.materials import *


class TestNetworkPredictStrategy(unittest.TestCase):
    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        network_predict_strategy = factory.make_neural_network_predict_strategy(neural_network)
        self.assertRaises(NotImplementedError, NetworkPredictStrategy.__call__, network_predict_strategy)

    def test_activate_neurons(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        network_predict_strategy = factory.make_neural_network_predict_strategy(neural_network)
        self.assertRaises(NotImplementedError, NetworkPredictStrategy.activate_neurons, network_predict_strategy)


class TestMlpPredictStrategy(unittest.TestCase):
    """ Unit tests for MLpPredictStrategy
    """

    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        predict_strategy = MlpNetworkPredictStrategy(neural_network)
        self.assertEqual(predict_strategy.neural_network, neural_network)

    def test_call_wrong_dimensions(self):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        predict_strategy = neural_network.predict_strategy
        input_data = np.random.rand(4, 4)
        self.assertRaises(ValueError, predict_strategy, input_data)

    def test_call__(self):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        predict_strategy = neural_network.predict_strategy
        input_data = np.random.rand(4, 3)
        result = np.zeros((4, 1))
        for row, data in enumerate(input_data):
            neural_network.read(data)
            neural_network.predict_strategy.activate_neurons()
            result[row, :] = neural_network.inspect_output()
        self.assertTrue((neural_network(input_data) == result).all)

    def test_activate_neurons(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        input_data = np.random.rand(2, 3)
        neural_network.read(input_data[1, :])
        neural_network.predict_strategy.activate_neurons()
        input_layer = neural_network.layers[0]
        hidden_layer = neural_network.layers[1]
        output_layer = neural_network.layers[2]
        neuron_a = hidden_layer[0]
        neuron_b = hidden_layer[1]
        neuron_c = output_layer[0]
        a = neuron_a()
        b = neuron_b()
        c = neuron_c()
        result = c
        self.assertEqual(neural_network.inspect_output(), [result])


if __name__ == '__main__':
    unittest.main()
