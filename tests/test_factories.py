import unittest

from amore.factories import *
from amore.materials import *
from amore.network_fit_strategies import *
from amore.network_predict_strategies import *
from amore.neuron_fit_strategies import *
from amore.neuron_predict_strategies import *


class TestNeuralFactory(unittest.TestCase):

    def test_make_neuron_fit_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        self.assertRaises(NotImplementedError, MaterialsFactory.make_neuron_fit_strategy, factory, factory)


class MlpNeuralNetworkFactory(unittest.TestCase):
    def test_make_primitive_connection(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        connection = factory.make_primitive_connection(neuron)
        self.assertTrue(isinstance(connection, MlpConnection))

    def test_make_primitive_container(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        connection = factory.make_primitive_connection(neuron)
        self.assertTrue(isinstance(connection, MlpConnection))

    def test_make_primitive_neuron(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        self.assertTrue(isinstance(neuron, MlpNeuron))

    def test_make_primitive_neural_network(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertTrue(isinstance(neural_network, MlpNetwork))

    def test_make_neural_network_builder(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        builder = factory.make_neural_network_builder()
        self.assertTrue(isinstance(builder, MlpNetworkBuilder))

    def test_make_activation_function(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        activation_function = factory.make_activation_function('tanh')
        self.assertTrue(activation_function is activation_functions_set['tanh'])

    def test_make_cost_function(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        cost_function = factory.make_cost_function('adaptLMS')
        self.assertTrue(cost_function is cost_functions_set['adaptLMS'])


class TestAdaptiveGradientDescentFactory(unittest.TestCase):
    """ Test for MlpFactory, a simple factories for simple multilayer feed forward networks"""

    def test_make_primitive_connection(self):
        """  MlpFactory unit test """
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        connection = factory.make_primitive_connection(neuron)
        self.assertTrue(connection.neuron is neuron)

    def test_make_primitive_container(self):
        """  MlpFactory unit test """
        factory = AdaptiveGradientDescentMaterialsFactory()
        container = factory.make_primitive_container()
        self.assertTrue(isinstance(container, type([])))
        self.assertEqual(len(container), 0)

    def test_make_primitive_neuron(self):
        """  MlpFactory unit test """
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        self.assertTrue(isinstance(neuron, type(MlpNeuron(neural_network))))

    def test_make_primitive_neural_network(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertEqual(type(neural_network), type(MlpNetwork(factory)))

    def test_make_neural_network_builder(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neural_network_builder = factory.make_neural_network_builder()
        self.assertEqual(type(neural_network_builder), type(MlpNetworkBuilder()))

    def test_make_neuron_predict_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = factory.make_primitive_neuron(neural_network)
        predict_strategy = factory.make_neuron_predict_strategy(neuron)
        self.assertTrue(type(predict_strategy), type(MlpNeuronPredictStrategy))

    def test_make_neuron_fit_strategy_neuron_in_output_layer(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
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
        factory = AdaptiveGradientDescentMaterialsFactory()
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
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        predict_strategy = MlpNetworkPredictStrategy(neural_network)
        self.assertTrue(type(predict_strategy), type(MlpNetworkPredictStrategy(neural_network)))

    def test_make_neural_network_fit_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_neural_network()
        fit_strategy = AdaptiveGradientDescentNetworkFitStrategy(neural_network)
        self.assertTrue(type(fit_strategy), type(AdaptiveGradientDescentNetworkFitStrategy(neural_network)))

    def test_make_activation_function(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        test_function = factory.make_activation_function('tanh')
        self.assertEqual(test_function(0.5), activation_functions_set['tanh'](0.5))

    def test_make_cost_function(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        test_function = factory.make_cost_function('adaptLMS')
        self.assertEqual(test_function(0.1, 0.6), cost_functions_set['adaptLMS'](0.1, 0.6))


if __name__ == '__main__':
    unittest.main()
