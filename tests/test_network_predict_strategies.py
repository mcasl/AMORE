import unittest
from amore import *


class TestNetworkPredictStrategy(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        network_predict_strategy = factory.make_neural_network_predict_strategy(neural_network)
        self.assertEqual(network_predict_strategy.neural_network, neural_network)

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

    def test__call__(self):
        neural_network = mlp_network([3, 5, 1], 'tanh', 'tanh')
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        # Prepare ad-hoc input_layer
        input_layer = factory.make_primitive_container()
        for dummy in range(5):
            neuron = factory.make_primitive_neuron(neural_network)
            neuron.output = random.random()
            input_layer.append(neuron)

        # Setup test neuron
        test_neuron = factory.make_primitive_neuron(neural_network)
        for neuron in input_layer:
            connection = factory.make_primitive_connection(neuron)
            connection.weight = random.random()
            test_neuron.connections.append(connection)

        # Expected result calculation
        accumulator = test_neuron.bias
        for connection in test_neuron.connections:
            accumulator += connection.neuron.output * connection.weight
        expected_result = test_neuron.activation_function(accumulator)
        self.assertEqual(test_neuron.predict_strategy(), expected_result)


if __name__ == '__main__':
    unittest.main()
