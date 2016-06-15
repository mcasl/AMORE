import unittest
from pyAmore.baseline.interface import *


class TestInterface(unittest.TestCase):
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
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['0', '1', '2'],
                                               ['3', '4', '5', '6', '7'],
                                               ['3', '4', '5', '6', '7']])
        neuron_labels = []
        for layer in neural_network.layers:
            neuron_labels.extend([neuron.label for neuron in layer])
        self.assertEqual(neuron_labels, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    def test_fit_adaptive_gradient_descent(self):
        random.seed(2016)
        np.random.seed(2016)
        input_data = np.random.rand(100, 3)
        target_data = input_data ** 2
        learning_rate = 0.25
        step_length = 100
        number_of_steps = 10

        neural_network = mlp_network([3, 4, 1], 'tanh', 'identity')
        original_weights = network_weights(neural_network)

        expected_original_weights = [np.array([[0.26540605, -0.1721736, 0.12930261, 0.30574679],
                                               [-0.29559449, -0.14375558, -0.30290563, -0.00934439],
                                               [-0.01905477, -0.14193801, -0.24395529, -0.20445159],
                                               [-0.20626199, -0.23438589, -0.03327293, 0.14908032]]),
                                     np.array([[-0.29559326, 0.09108178, -0.18177749, 0.07662673, 0.25499338]])]

        self.assertTrue((expected_original_weights[0] == original_weights[0].round(8)).all())
        self.assertTrue((expected_original_weights[1] == original_weights[1].round(8)).all())

        fit_adaptive_gradient_descent(neural_network,
                                      input_data,
                                      target_data,
                                      learning_rate,
                                      step_length,
                                      number_of_steps)

        trained_weights = network_weights(neural_network)

        expected_trained_weights = [np.array([[0.38062303, -0.05418239, 0.01853164, 0.94158979],
                                              [-0.13731969, -0.21905795, -0.37320761, 0.55868609],
                                              [-0.42297188, -0.13782546, -0.08766303, -0.85411441],
                                              [-2.34937046, -0.03954766, -0.0487216, 2.70076885]]),
                                    np.array([[0.5815685, 0.09805496, -0.55547039, -2.03304081, 1.13027785]])]

        self.assertTrue((expected_trained_weights[0] == trained_weights[0].round(8)).all())
        self.assertTrue((expected_trained_weights[1] == trained_weights[1].round(8)).all())


if __name__ == '__main__':
    unittest.main()
