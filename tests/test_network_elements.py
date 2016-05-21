import random
import unittest
import numpy as np

from hypothesis import given, strategies as st
from amore.factories import *
from amore.materials import *
from amore.network_fit_strategies import *
from amore.network_predict_strategies import *
from amore.neuron_predict_strategies import *


class TestConnection(unittest.TestCase):
    """ Tests for Connection class """

    def test_init_where_weight_is_default(self):
        """ Connection constructor test. weight attribute initialization is checked
        """
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = MlpNeuron(neural_network)
        connection = Connection(neuron)
        self.assertEqual(connection.weight, 0.0)

    @given(weight=st.floats())
    def test_init_where_weight_is_given(self, weight):
        """ Connection constructor test. weight attribute initialization is checked
        """
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = MlpNeuron(neural_network)
        connection = Connection(neuron, weight)
        self.assertTrue(connection.weight is weight)


class TestNeuron(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = MlpNeuron(neural_network)
        self.assertEqual(neuron.label, None)
        self.assertEqual(neuron.neural_network, neural_network)
        self.assertTrue(isinstance(neuron.predict_strategy, AdaptiveGradientDescentNeuronPredictStrategy))

    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = MlpNeuron(neural_network)
        self.assertRaises(NotImplementedError, Neuron.__call__, neuron)


class TestMlpNeuron(unittest.TestCase):
    """ Tests for SimpleNeuron class, a simple multilayer feed forward neural network neuron
    """

    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neuron = MlpNeuron(neural_network)
        attributes = [None, 0.0, Container()]
        neuron_attributes = [neuron.label, neuron.output, neuron.connections]
        self.assertEqual(neuron_attributes, attributes)

    def test_fit(self):
        pass  # TODO: test

    def test___call__(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        # Prepare ad-hoc input_layer
        input_layer = factory.make_primitive_container()
        for dummy in range(5):
            neuron = factory.make_primitive_neuron(neural_network)
            neuron.output = random.random()
            input_layer.append(neuron)

        # Setup test neuron
        test_neuron = factory.make_primitive_neuron(neural_network)
        for neuron in input_layer:
            connection = factory.make_primitive_connection(neuron)
            connection.weight = random.random()
            test_neuron.connections.append(connection)
        self.assertEqual(test_neuron(), test_neuron.predict_strategy())


class TestNeuralNetwork(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertEqual(neural_network.factory, factory)
        self.assertTrue(isinstance(neural_network.predict_strategy, AdaptiveGradientDescentNetworkPredictStrategy))
        self.assertTrue(isinstance(neural_network.fit_strategy, AdaptiveGradientDescentNetworkFitStrategy))

    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertRaises(NotImplementedError, NeuralNetwork.__call__, neural_network)

    def test_read_input_data(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        data = None
        self.assertRaises(NotImplementedError, NeuralNetwork.read, neural_network, data)

    def test_inspect_output(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertRaises(NotImplementedError, NeuralNetwork.inspect_output, neural_network)

        # def test_shape(self):
        #     factory = AdaptiveGradientDescentFactory()
        #     neural_network = factory.make_primitive_neural_network()
        #     self.assertRaises(NotImplementedError, NeuralNetwork.shape, neural_network)


class TestSimpleNeuralNetwork(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertEqual(neural_network.layers, factory.make_primitive_container())

    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        builder = factory.make_neural_network_builder()
        neural_network = builder.create_neural_network(factory, [3, 6, 2], 'tanh', 'identity')
        input_data = np.ones((100, 3))
        self.assertTrue((neural_network(input_data) == neural_network.predict_strategy(input_data)).all)

    def test_fit(self):
        pass  # TODO: test

    def test_predict(self):
        pass  # TODO: test

    def test_insert_input_data(self, shape=(40, 3, 2)):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neural_network_builder = factory.make_neural_network_builder()
        neural_network = neural_network_builder.create_neural_network(factory, shape, 'tanh', 'identity')
        sample_data = [random.random() for dummy in range(shape[0])]
        neural_network.read(sample_data)
        result = np.asarray([neuron.output for neuron in neural_network.layers[0]])
        self.assertTrue((result == sample_data).all)

    def test_write_targets_in_output_layer(self, shape=(4, 6, 78)):
        factory = AdaptiveGradientDescentFactory()
        neural_creator = factory.make_neural_network_builder()
        neural_network = neural_creator.create_neural_network(factory, shape, 'tanh', 'identity')
        sample_data = [random.random() for dummy in range(shape[-1])]
        neural_network.fit_strategy.write_targets_in_output_layer(sample_data)
        result = [neuron.fit_strategy.target for neuron in neural_network.layers[-1]]
        self.assertEqual(result, sample_data)

    def test_shape(self, shape=(40, 52, 1, 7, 4)):
        factory = AdaptiveGradientDescentFactory()
        neural_creator = factory.make_neural_network_builder()
        neural_network = neural_creator.create_neural_network(factory, shape, 'tanh', 'identity')
        self.assertEqual(neural_network.shape, list(shape))

    def test_single_pattern_forward_action(self):
        pass

    def test_single_pattern_backward_action(self):
        pass

    def test_read_output_layer(self, shape=(4, 6, 78)):
        factory = AdaptiveGradientDescentFactory()
        neural_creator = factory.make_neural_network_builder()
        neural_network = neural_creator.create_neural_network(factory, shape, 'tanh', 'identity')
        sample_data = [random.random() for dummy in range(shape[-1])]
        for neuron, output_value in zip(neural_network.layers[-1], sample_data):
            neuron.output = output_value
        result = neural_network.inspect_output()
        self.assertEqual(result, sample_data)


if __name__ == '__main__':
    unittest.main()
