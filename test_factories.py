import unittest

from amore import *
from container import *
from neuron_predict_strategies import *


class TestAdaptiveGradientDescentFactory(unittest.TestCase):
    """ Test for MlpFactory, a simple factory for simple multilayer feed forward networks"""

    def test_make_primitive_connection(self):
        """  MlpFactory unit test """
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        connection = factory.make_primitive_connection(neuron)
        self.assertTrue(connection.neuron is neuron)

    def test_make_primitive_container(self):
        """  MlpFactory unit test """
        factory = AdaptiveGradientDescentFactory()
        container = factory.make_primitive_container()
        self.assertTrue(isinstance(container, type(Container())))
        self.assertEqual(len(container), 0)

    def test_make_primitive_neuron(self):
        """  MlpFactory unit test """
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        self.assertTrue(isinstance(neuron, type(MlpNeuron(neural_network))))

    def test_make_primitive_layer(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        layer = factory.make_primitive_layer(20, neural_network)
        self.assertEqual(len(layer), 20)

    def test_make_primitive_neural_network(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertEqual(type(neural_network), type(MlpNeuralNetwork(factory)))

    def test_make_neural_network_builder(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neural_network_builder = factory.make_neural_network_builder()
        self.assertEqual(type(neural_network_builder), type(MlpNeuralNetworkBuilder()))

    def test_make_neuron_predict_strategy(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        predict_strategy = AdaptiveGradientDescentNeuronPredictStrategy(neuron)
        self.assertTrue(type(predict_strategy), type(AdaptiveGradientDescentNeuronPredictStrategy))

    def test_make_neuron_fit_strategy_neuron_in_output_layer(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        neural_network.layers = factory.make_primitive_container()
        neural_network.layers.append(factory.make_primitive_container())
        neural_network.layers.append(factory.make_primitive_container())
        neural_network.layers.append(factory.make_primitive_container())
        neural_network.layers[-1].append(neuron)
        fit_strategy = factory.make_neuron_fit_strategy(neuron)
        self.assertTrue(type(fit_strategy), type(AdaptiveGradientDescentOutputNeuronFitStrategy(neuron)))

    def test_make_neuron_fit_strategy_neuron_in_hidden_layer(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        neural_network.layers = factory.make_primitive_container()
        neural_network.layers.append(factory.make_primitive_container())
        neural_network.layers.append(factory.make_primitive_container())
        neural_network.layers.append(factory.make_primitive_container())
        neural_network.layers[1].append(neuron)
        fit_strategy = factory.make_neuron_fit_strategy(neuron)
        self.assertTrue(type(fit_strategy), type(AdaptiveGradientDescentHiddenNeuronFitStrategy(neuron)))

    def test_make_neural_network_predict_strategy(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        predict_strategy = AdaptiveGradientDescentNetworkPredictStrategy(neural_network)
        self.assertTrue(type(predict_strategy), type(AdaptiveGradientDescentNetworkPredictStrategy(neural_network)))

    def test_make_neural_network_fit_strategy(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        fit_strategy = AdaptiveGradientDescentNetworkFitStrategy(neural_network)
        self.assertTrue(type(fit_strategy), type(AdaptiveGradientDescentNetworkFitStrategy(neural_network)))

    def test_make_activation_function(self):
        factory = AdaptiveGradientDescentFactory()
        test_function = factory.make_activation_function('tanh')
        self.assertEqual(test_function(0.5), activation_functions_set['tanh'](0.5))

    def test_make_cost_function(self):
        factory = AdaptiveGradientDescentFactory()
        test_function = factory.make_cost_function('LMS')
        self.assertEqual(test_function(0.1, 0.6), cost_functions_set['LMS'](0.1, 0.6))


if __name__ == '__main__':
    unittest.main()
