import unittest

from amore.interface import *


class TestAmore(unittest.TestCase):
    def test_mlp_network(self):
        neural_network = mlp_network(layers_size=[3, 5, 2],
                                     hidden_layers_activation_function_name='tanh',
                                     output_layer_activation_function_name='identity'
                                     )
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
                                               [3, 4, 5, 6, 7]])
        neuron_labels = []
        for layer in neural_network.layers:
            neuron_labels.extend([neuron.label for neuron in layer])
        self.assertEqual(neuron_labels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == '__main__':
    unittest.main()
