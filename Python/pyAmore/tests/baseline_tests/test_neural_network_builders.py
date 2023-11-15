import unittest

from pyAmore.baseline.factories import *

from pyAmore.baseline.builders import NetworkBuilder, MlpNetworkBuilder


class TestNeuralNetworkBuilder(unittest.TestCase):
    def test_create_neural_network(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        builder = factory.make_network_builder()
        self.assertRaises(NotImplementedError, NetworkBuilder.create_neural_network, builder)


class TestMlpNeuralBuilder(unittest.TestCase):
    """ Unit tests for SimpleNeuralCreator class, the builders of SimpleNeuralNetworks
    """

    def test_create_neural_network(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        builder = factory.make_network_builder()
        neural_network = builder.create_neural_network([3, 5, 2], 'tanh', 'tanh')
        self.assertTrue(isinstance(neural_network, MlpNetwork))

    def test_set_neurons_fit_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        builder = factory.make_network_builder()
        neural_network = builder.create_neural_network([3, 4, 5], 'tanh', 'tanh')
        builder.set_neurons_fit_strategy(neural_network)
        result = []
        for layer in neural_network.layers:
            result.extend(neuron.fit_strategy for neuron in layer)
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

    def test_connect_network_layers(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        builder = factory.make_network_builder()
        neural_network = factory.make_primitive_network()
        builder.create_primitive_layers(neural_network, [3, 5, 2])
        builder.connect_network_layers(neural_network)
        builder.initialize_network(neural_network)
        labels = [
            [neuron.label for neuron in layer] for layer in neural_network.layers
        ]
        self.assertEqual(labels, [['0', '1', '2'], ['3', '4', '5', '6', '7'], ['8', '9']])

        network_connections = []
        for layer in neural_network.layers:
            network_connections.extend(
                [origin.neuron.label for origin in neuron.connections]
                for neuron in layer
            )
        self.assertEqual(network_connections, [[],
                                               [],
                                               [],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['3', '4', '5', '6', '7'],
                                               ['3', '4', '5', '6', '7']
                                               ])

    def test_initialize_network_empty_layers(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        builder = factory.make_network_builder()
        self.assertRaises(ValueError, builder.initialize_network, neural_network)

    def test_initialize_network_where_weights_and_biases_change(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        builder = factory.make_network_builder()
        builder.create_primitive_layers(neural_network, [3, 5, 2])
        builder.connect_network_layers(neural_network)

        pre_initialized_weights = []
        pre_initialized_biases = []
        for layer in neural_network.layers:
            for neuron in layer:
                pre_initialized_biases.append(neuron.bias)
                pre_initialized_weights.extend(
                    connection.weight for connection in neuron.connections
                )
        builder.initialize_network(neural_network)

        post_initialized_weights = []
        post_initialized_biases = []
        for layer in neural_network.layers:
            for neuron in layer:
                post_initialized_biases.append(neuron.bias)
                post_initialized_weights.extend(
                    connection.weight for connection in neuron.connections
                )
        self.assertNotEqual(pre_initialized_weights, post_initialized_weights)
        self.assertNotEqual(pre_initialized_biases, post_initialized_biases)

    def test_create_primitive_layers(self):
        """ TestSimpleNeuralCreator Unit tests """
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        builder = factory.make_network_builder()
        builder.create_primitive_layers(neural_network, [3, 5, 2])
        self.assertEqual(len(neural_network.layers), 3)
        self.assertEqual(list(map(len, neural_network.layers)), [3, 5, 2])


if __name__ == '__main__':
    unittest.main()
