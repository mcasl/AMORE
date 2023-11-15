import unittest
from math import isnan
from hypothesis import given, assume, strategies as st
from pyAmore.cython.interface import mlp_network
from pyAmore.cython.materials import *
from pyAmore.cython.network_fit_strategies import *
from pyAmore.cython.network_predict_strategies import *
from pyAmore.cython.neuron_predict_strategies import *

from pyAmore.cython.factories import *


class TestMlpContainer(unittest.TestCase):
    """ Tests for Container class, currently a list subclass"""

    def test_init_where_iterable_is_default(self):
        container = MlpContainer()
        self.assertEqual(container, [])

    def test_init_where_iterable_is_sequence(self):
        container = MlpContainer([2, 5, 67])
        self.assertEqual(container, [2, 5, 67])


class TestConnection(unittest.TestCase):
    """ Tests for Connection class """

    def test_init_where_weight_is_default(self):
        """ Connection constructor test. weight attribute initialization is checked
        """
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = MlpNeuron(neural_network)
        connection = MlpConnection(neuron)
        self.assertEqual(connection.weight, 0.0)
        self.assertTrue(connection.neuron is neuron)

    @given(weight=st.floats())
    def test_init_where_weight_is_given(self, weight):
        """ Connection constructor test. weight attribute initialization is checked
        """
        assume(not isnan(weight))
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = MlpNeuron(neural_network)
        connection = MlpConnection(neuron, weight)
        self.assertEquals(connection.weight, weight)
        self.assertTrue(connection.neuron is neuron)


class TestNeuron(unittest.TestCase):
    def test_call(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = MlpNeuron(neural_network)
        self.assertRaises(NotImplementedError, Neuron.__call__, neuron)


class TestMlpNeuron(unittest.TestCase):
    """ Tests for SimpleNeuron class, a simple multilayer feed forward neural network neuron
    """

    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = MlpNeuron(neural_network)
        self.assertEquals(neuron.label, None)
        self.assertEquals(neuron.output, 0.0)
        self.assertEquals(neuron.predict_strategy, None)
        self.assertEquals(neuron.fit_strategy, None)
        self.assertEquals(neuron.connections, [])
        self.assertEquals(neuron.bias, 0.0)
        self.assertTrue(neuron.activation_function is activation_functions_set['default'])

    def test_call(self):
        neural_network = mlp_network([4, 3, 2], 'tanh', 'identity')
        test_neuron = neural_network.layers[1][1]
        self.assertEqual(test_neuron(), test_neuron.predict_strategy.predict())


class TestNetwork(unittest.TestCase):
    def test_init(self):
        neural_network = mlp_network([4, 3, 2], 'tanh', 'identity')
        self.assertTrue(isinstance(neural_network.factory, AdaptiveGradientDescentMaterialsFactory))
        self.assertTrue(isinstance(neural_network.predict_strategy, MlpNetworkPredictStrategy))
        self.assertTrue(isinstance(neural_network.fit_strategy, AdaptiveNetworkFitStrategy))

    def test_call(self):
        neural_network = mlp_network([4, 3, 2], 'tanh', 'identity')
        self.assertRaises(NotImplementedError, Network.__call__, neural_network)

    def test_poke_inputs(self):
        neural_network = mlp_network([4, 3, 2], 'tanh', 'identity')
        data = None
        self.assertRaises(NotImplementedError, Network.poke_inputs, neural_network, data)

    def test_pick_outputs(self):
        neural_network = mlp_network([4, 3, 2], 'tanh', 'identity')
        self.assertRaises(NotImplementedError, Network.pick_outputs, neural_network)

    def test_shape(self):
        neural_network = mlp_network([4, 3, 2], 'tanh', 'identity')
        self.assertRaises(NotImplementedError, Network.shape.__get__, neural_network)


class TestMlpNetwork(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = MlpNetwork(factory)
        self.assertEqual(neural_network.layers, factory.make_primitive_container())

    def test_call(self, input_size=100, shape=(4, 3, 2)):
        neural_network = mlp_network(list(shape), 'tanh', 'identity')
        input_data = np.random.rand(input_size, shape[0])
        self.assertTrue((neural_network(input_data) == neural_network.predict_strategy(input_data)).all)

    def test_poke_inputs(self, shape=(40, 3, 2)):
        # Stage 1: Poke data
        neural_network = mlp_network(list(shape), 'tanh', 'identity')
        sample_data = np.random.rand(shape[0])
        neural_network.poke_inputs(sample_data)
        # Stage 2: Verify previous operation
        input_layer = neural_network.layers[0]
        observed_data = np.asarray([neuron.output for neuron in input_layer])
        self.assertTrue((observed_data == sample_data).all)

    def test_pick_outputs(self, shape=(4, 6, 78)):
        # Stage 1: Write data in last layer neuron outputs
        neural_network = mlp_network(list(shape), 'tanh', 'identity')
        sample_data = np.random.rand(shape[-1])
        last_layer = neural_network.layers[-1]
        for neuron, output_value in zip(last_layer, sample_data):
            neuron.output = output_value
        # Stage 2: Check pick_outputs
        observed_data = np.asarray(neural_network.pick_outputs())
        self.assertTrue((observed_data == sample_data).all)

    def test_shape(self, shape=(40, 52, 1, 7, 4)):
        neural_network = mlp_network(list(shape), 'tanh', 'identity')
        self.assertEqual(neural_network.shape, list(shape))


if __name__ == '__main__':
    unittest.main()
