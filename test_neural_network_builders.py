import unittest
from amore import *


class TestMlpNeuralBuilder(unittest.TestCase):
    """ Unit tests for SimpleNeuralCreator class, the builder of SimpleNeuralNetworks
    """

    def test_create_primitive_layers(self):
        """ TestSimpleNeuralCreator Unit tests """
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        MlpNeuralNetworkBuilder.create_primitive_layers(factory, neural_network, [3, 5, 2])
        self.assertEqual(len(neural_network.layers), 3)
        self.assertEqual(list(map(len, neural_network.layers)), [3, 5, 2])

    def test_connect_and_initialize_network(self):
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


if __name__ == '__main__':
    unittest.main()
