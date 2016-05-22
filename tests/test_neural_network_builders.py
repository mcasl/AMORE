import unittest

from amore.factories import *
from amore.builders import NeuralNetworkBuilder, MlpNeuralNetworkBuilder


class TestNeuralNetworkBuilder(unittest.TestCase):
    def test_create_neural_network(self):
        factory = AdaptiveGradientDescentFactory()
        builder = factory.make_neural_network_builder()
        self.assertRaises(NotImplementedError, NeuralNetworkBuilder.create_neural_network, builder)


class TestMlpNeuralBuilder(unittest.TestCase):
    """ Unit tests for SimpleNeuralCreator class, the builders of SimpleNeuralNetworks
    """

    def test_create_neural_network(self):
        factory = AdaptiveGradientDescentFactory()
        builder = factory.make_neural_network_builder()
        neural_network = builder.create_neural_network(factory, [3, 5, 2], 'tanh', 'tanh')
        self.assertTrue(isinstance(neural_network, MlpNeuralNetwork))

    def test_connect_network_layers(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        MlpNeuralNetworkBuilder.create_primitive_layers(factory, neural_network, [3, 5, 2])
        MlpNeuralNetworkBuilder.connect_network_layers(factory, neural_network)
        MlpNeuralNetworkBuilder.initialize_network(neural_network)
        labels = []
        for layer in neural_network.layers:
            labels.append([neuron.label for neuron in layer])
        self.assertEqual(labels, [[0, 1, 2], [3, 4, 5, 6, 7], [8, 9]])

        network_connections = []
        for layer in neural_network.layers:
            for neuron in layer:
                network_connections.append([origin.neuron.label for origin in neuron.connections])
        self.assertEqual(network_connections, [[],
                                               [],
                                               [],
                                               [0, 1, 2],
                                               [0, 1, 2],
                                               [0, 1, 2],
                                               [0, 1, 2],
                                               [0, 1, 2],
                                               [3, 4, 5, 6, 7],
                                               [3, 4, 5, 6, 7]
                                               ])

    def test_initialize_network_empty_layers(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertRaises(ValueError, MlpNeuralNetworkBuilder.initialize_network, neural_network)

    def test_initialize_network_where_weights_and_biases_change(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        MlpNeuralNetworkBuilder.create_primitive_layers(factory, neural_network, [3, 5, 2])
        MlpNeuralNetworkBuilder.connect_network_layers(factory, neural_network)

        pre_initialized_weights = []
        pre_initialized_biases = []
        for layer in neural_network.layers:
            for neuron in layer:
                pre_initialized_biases.append(neuron.bias)
                for connection in neuron.connections:
                    pre_initialized_weights.append(connection.weight)

        MlpNeuralNetworkBuilder.initialize_network(neural_network)

        post_initialized_weights = []
        post_initialized_biases = []
        for layer in neural_network.layers:
            for neuron in layer:
                post_initialized_biases.append(neuron.bias)
                for connection in neuron.connections:
                    post_initialized_weights.append(connection.weight)

        self.assertNotEqual(pre_initialized_weights, post_initialized_weights)
        self.assertNotEqual(pre_initialized_biases, post_initialized_biases)

    def test_create_primitive_layers(self):
        """ TestSimpleNeuralCreator Unit tests """
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        MlpNeuralNetworkBuilder.create_primitive_layers(factory, neural_network, [3, 5, 2])
        self.assertEqual(len(neural_network.layers), 3)
        self.assertEqual(list(map(len, neural_network.layers)), [3, 5, 2])


if __name__ == '__main__':
    unittest.main()
