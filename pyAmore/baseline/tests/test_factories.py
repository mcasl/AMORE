import unittest

from pyAmore.baseline.interface import mlp_network
from pyAmore.baseline.materials import *
from pyAmore.baseline.network_fit_strategies import *
from pyAmore.baseline.network_predict_strategies import *
from pyAmore.baseline.neuron_fit_strategies import *
from pyAmore.baseline.neuron_predict_strategies import *

from pyAmore.baseline.factories import *


class TestFactory(unittest.TestCase):
    def test_init(self):
        factory = Factory()
        expected_class_names = {}
        self.assertEqual(factory.class_names, expected_class_names)


class TestMaterialsFactory(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        factory.class_names.clear()
        MaterialsFactory.__init__(factory)
        expected_class_names = {}
        self.assertEqual(factory.class_names, expected_class_names)

    def test_make_network_builder(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        builder = factory.make_network_builder()
        self.assertTrue(isinstance(builder, MlpNetworkBuilder))

    def test_make_primitive_neural_network(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        self.assertTrue(isinstance(neural_network, MlpNetwork))

    def test_make_primitive_container(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        container = factory.make_primitive_container()
        self.assertTrue(isinstance(container, MlpContainer))

    def test_make_primitive_neuron(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = factory.make_primitive_neuron(neural_network)
        self.assertTrue(isinstance(neuron, MlpNeuron))

    def test_make_primitive_connection(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = factory.make_primitive_neuron(neural_network)
        connection = factory.make_primitive_connection(neuron)
        self.assertTrue(isinstance(connection, MlpConnection))

    def test_make_network_predict_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        predict_strategy = factory.make_network_predict_strategy(neural_network)
        self.assertTrue(isinstance(predict_strategy, MlpNetworkPredictStrategy))

    def test_make_network_fit_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        fit_strategy = factory.make_network_fit_strategy(neural_network)
        self.assertTrue(isinstance(fit_strategy, AdaptiveNetworkFitStrategy))

    def test_make_neuron_predict_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = factory.make_primitive_neuron(neural_network)
        predict_strategy = factory.make_neuron_predict_strategy(neuron)
        self.assertTrue(isinstance(predict_strategy, MlpNeuronPredictStrategy))

    def test_make_neuron_fit_strategy(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        self.assertRaises(NotImplementedError, MaterialsFactory.make_neuron_fit_strategy, factory, factory)

    def test_make_activation_function(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        activation_function = factory.make_activation_function('default')
        self.assertTrue(activation_function is activation_functions_set['default'])

    def test_make_cost_function(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        cost_function = factory.make_cost_function('default')
        self.assertTrue(cost_function is cost_functions_set['default'])


class TestMlpMaterialsFactory(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        factory.class_names.clear()
        MlpMaterialsFactory.__init__(factory)
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                }
        self.assertEqual(factory.class_names, expected_class_names)

    def test_make_neuron_fit_strategy_neuron_in_output_layer(self, output_layer_size=20):
        neural_network = mlp_network([3, 3, output_layer_size], 'default', 'default')
        factory = AdaptiveGradientDescentMaterialsFactory()
        for output_neuron in neural_network.layers[-1]:
            fit_strategy = factory.make_neuron_fit_strategy(output_neuron)
            self.assertTrue(isinstance(fit_strategy, AdaptiveGradientDescentOutputNeuronFitStrategy))

    def test_make_neuron_fit_strategy_neuron_in_hidden_layer(self, hidden_layer_size=20):
        neural_network = mlp_network([3, hidden_layer_size, 2], 'default', 'default')
        factory = AdaptiveGradientDescentMaterialsFactory()
        for hidden_neuron in neural_network.layers[1]:
            fit_strategy = factory.make_neuron_fit_strategy(hidden_neuron)
            self.assertTrue(isinstance(fit_strategy, AdaptiveGradientDescentHiddenNeuronFitStrategy))


class TestAdaptiveMaterialsFactory(unittest.TestCase):
    """ Test for MlpFactory, a simple factories for simple multilayer feed forward networks"""

    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        factory.class_names.clear()
        AdaptiveMaterialsFactory.__init__(factory)
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                'NetworkFitStrategy': 'AdaptiveNetworkFitStrategy',
                                }

        self.assertEqual(factory.class_names, expected_class_names)


class TestBatchMaterialsFactory(unittest.TestCase):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

    def test_init(self):
        factory = BatchGradientDescentMaterialsFactory()
        factory.class_names.clear()
        BatchMaterialsFactory.__init__(factory)
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                'NetworkFitStrategy': 'BatchNetworkFitStrategy',
                                }

        self.assertEqual(factory.class_names, expected_class_names)


class TestAdaptiveGradientDescentMaterialsFactory(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                'NetworkFitStrategy': 'AdaptiveNetworkFitStrategy',
                                'OutputNeuronFitStrategy': 'AdaptiveGradientDescentOutputNeuronFitStrategy',
                                'HiddenNeuronFitStrategy': 'AdaptiveGradientDescentHiddenNeuronFitStrategy',
                                }
        self.assertEqual(factory.class_names, expected_class_names)


class TestAdaptiveGradientDescentWithMomentumMaterialsFactory(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentWithMomentumMaterialsFactory()
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                'NetworkFitStrategy': 'AdaptiveNetworkFitStrategy',
                                'OutputNeuronFitStrategy': 'AdaptiveGradientDescentWithMomentumOutputNeuronFitStrategy',
                                'HiddenNeuronFitStrategy': 'AdaptiveGradientDescentWithMomentumHiddenNeuronFitStrategy',
                                }
        self.assertEqual(factory.class_names, expected_class_names)


class TestBatchGradientDescentMaterialsFactory(unittest.TestCase):
    def test_init(self):
        factory = BatchGradientDescentMaterialsFactory()
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                'NetworkFitStrategy': 'BatchNetworkFitStrategy',
                                'OutputNeuronFitStrategy': 'BatchGradientDescentOutputNeuronFitStrategy',
                                'HiddenNeuronFitStrategy': 'BatchGradientDescentHiddenNeuronFitStrategy',
                                }
        self.assertEqual(factory.class_names, expected_class_names)


class TestBatchGradientDescentWithMomentumMaterialsFactory(unittest.TestCase):
    def test_init(self):
        factory = BatchGradientDescentWithMomentumMaterialsFactory()
        expected_class_names = {'Connection': 'MlpConnection',
                                'Container': 'MlpContainer',
                                'Neuron': 'MlpNeuron',
                                'Network': 'MlpNetwork',
                                'NeuronPredictStrategy': 'MlpNeuronPredictStrategy',
                                'NetworkPredictStrategy': 'MlpNetworkPredictStrategy',
                                'NetworkBuilder': 'MlpNetworkBuilder',
                                'NetworkFitStrategy': 'BatchNetworkFitStrategy',
                                'OutputNeuronFitStrategy': 'BatchGradientDescentWithMomentumOutputNeuronFitStrategy',
                                'HiddenNeuronFitStrategy': 'BatchGradientDescentWithMomentumHiddenNeuronFitStrategy',
                                }
        self.assertEqual(factory.class_names, expected_class_names)

if __name__ == '__main__':
    unittest.main()
