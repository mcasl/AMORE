import unittest

from pyAmore.baseline.interface import *
from pyAmore.baseline.materials import *
from pyAmore.baseline.network_fit_strategies import *

from pyAmore.baseline.neuron_fit_strategies import *


class TestNetworkFitStrategies(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        fit_strategy = factory.make_network_fit_strategy(neural_network)
        self.assertEqual(fit_strategy.neural_network, neural_network)

    def test_call(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        fit_strategy = factory.make_network_fit_strategy(neural_network)
        self.assertRaises(NotImplementedError, NetworkFitStrategy.__call__, fit_strategy)


class TestMlpNetworkFitStrategy(unittest.TestCase):
    def test_init(self):
        neural_network = mlp_network([3, 4, 5], 'default', 'default')
        strategy = AdaptiveNetworkFitStrategy(neural_network)

    def test_call(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        fit_strategy = factory.make_network_fit_strategy(neural_network)
        self.assertRaises(NotImplementedError, MlpNetworkFitStrategy.__call__, fit_strategy)

    def test_poke_targets(self):
        # Stage 1: Poke data
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        targets = [1, 2, 3, 4, 5]
        neural_network.fit_strategy.poke_targets(targets)
        # Stage 2: Verify poked data
        output_layer = neural_network.layers[-1]
        observed_targets = [neuron.fit_strategy.target for neuron in output_layer]
        self.assertEqual(targets, observed_targets)

    def test_backpropagate(self):
        random.seed(2016)
        np.random.seed(2016)
        neural_network = mlp_network([3, 4, 5], 'tanh', 'tanh')
        target_data = np.random.rand(5, 1)
        neural_network.fit_strategy.poke_targets(target_data)
        neural_network.fit_strategy.backpropagate()
        trained_weights = network_weights(neural_network)
        expected_trained_weights = [np.array([[0.19601536, -0.12715863, 0.09549631, 0.20019443],
                                              [-0.218311, -0.10617053, -0.22371064, -0.03554474],
                                              [-0.01407288, -0.10482817, -0.18017292, -0.13489893],
                                              [-0.15233457, -0.17310545, -0.02457369, 0.07546265]]),
                                    np.array([[-0.21831009, 0.06726835, -0.13425157, 0.05659259, 0.27799563],
                                              [0.1426101, -0.24345775, 0.03557834, -0.15794492, -0.02228203],
                                              [0.06562893, -0.22314618, 0.1380508, -0.21051052, 0.09395455],
                                              [-0.2446657, 0.04345133, 0.18973865, -0.0777204, 0.1634457],
                                              [-0.07460417, -0.05715924, 0.01414722, -0.12829666, 0.1563569]])]
        self.assertTrue((expected_trained_weights[0] == trained_weights[0].round(8)).all())
        self.assertTrue((expected_trained_weights[1] == trained_weights[1].round(8)).all())


class TestAdaptiveGradientDescentNetworkFitStrategy(unittest.TestCase):
    def test_call(self):
        random.seed(2016)
        np.random.seed(2016)
        input_data = np.random.rand(100, 3)
        target_data = input_data ** 2
        neural_network = mlp_network([3, 4, 1], 'tanh', 'identity')
        original_weights = network_weights(neural_network)

        expected_original_weights = [np.array([[0.26540605, -0.1721736, 0.12930261, 0.30574679],
                                               [-0.29559449, -0.14375558, -0.30290563, -0.00934439],
                                               [-0.01905477, -0.14193801, -0.24395529, -0.20445159],
                                               [-0.20626199, -0.23438589, -0.03327293, 0.14908032]]),
                                     np.array([[-0.29559326, 0.09108178, -0.18177749, 0.07662673, 0.25499338]])]

        self.assertTrue((expected_original_weights[0] == original_weights[0].round(8)).all())
        self.assertTrue((expected_original_weights[1] == original_weights[1].round(8)).all())

        neural_network.fit_strategy(input_data, target_data)
        trained_weights = network_weights(neural_network)

        expected_trained_weights = [np.array([[0.15912053, -0.14916691, 0.15508282, 0.33510523],
                                              [-0.30474322, -0.14689775, -0.30254587, -0.01861767],
                                              [-0.14270636, -0.13942633, -0.23524681, -0.21674058],
                                              [-0.20194557, -0.23471339, -0.03015267, 0.14808039]]),
                                    np.array([[-0.16367889, -0.05745085, -0.19707385, -0.04459547, 0.26388345]])]
        self.assertTrue((expected_trained_weights[0] == trained_weights[0].round(8)).all())
        self.assertTrue((expected_trained_weights[1] == trained_weights[1].round(8)).all())
