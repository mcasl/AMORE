import random
import unittest

import numpy
from hypothesis import given, strategies as st

from amore import Connection, SimpleNeuron, MlpFactory, SimpleNeuralCreator
from container import Container


class TestConnection(unittest.TestCase):
    """ Tests for Connection class """

    @given(label=st.integers(), weight=st.floats())
    def test_init_where_label_is_given(self, label, weight):
        """ Connection constructor test. label attribute initialization is checked
        """
        my_neuron = SimpleNeuron()
        my_neuron.label = label
        my_connection = Connection(my_neuron, weight)
        self.assertTrue(my_connection.neuron.label is label)

    @given(label=st.integers(), weight=st.floats())
    def test_init_where_weight_is_given(self, label, weight):
        """ Connection constructor test. weight attribute initialization is checked
        """
        my_neuron = SimpleNeuron()
        my_neuron.label = label
        my_connection = Connection(my_neuron, weight)
        self.assertTrue(my_connection.weight is weight)

    def test_repr(self, label=12, weight=0.23):
        """ Connection repr test. Simple string check
        """
        my_neuron = SimpleNeuron()
        my_neuron.label = label
        my_connection = Connection(my_neuron, weight)
        self.assertEqual(repr(my_connection), '\nFrom:\t 12 \t Weight= \t 0.23')


class TestSimpleNeuron(unittest.TestCase):
    """ Tests for SimpleNeuron class, a simple multilayer feed forward neural network neuron
    """

    def test_init(self):
        neuron = SimpleNeuron()
        attributes = [None, 0.0, 0.0, 0.0, Container()]
        neuron_attributes = [neuron.label, neuron.output, neuron.target, neuron.induced_local_field, neuron.connections]
        self.assertEqual(neuron_attributes, attributes)


class TestSimpleNeuralNetwork(unittest.TestCase):
    def test_init(self):
        neural_factory = MlpFactory()
        neural_network = neural_factory.make_primitive_neural_network()
        self.assertEqual(neural_network.layers, neural_factory.make_container())

    def test_insert_input_data(self, shape=(40, 3, 2)):
        factory = MlpFactory()
        neural_creator = factory.make_neural_creator()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        sample_data = [random.random() for x in range(shape[0])]
        neural_network.insert_input_data(sample_data)
        result = numpy.asarray([neuron.output for neuron in neural_network.layers[0]])
        self.assertTrue((result == sample_data).all)

    def test_write_targets_in_output_layer(self, shape=(4, 6, 78)):
        factory = MlpFactory()
        neural_creator = factory.make_neural_creator()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        sample_data = [random.random() for x in range(shape[-1])]
        neural_network.write_targets_in_output_layer(sample_data)
        result = [neuron.target for neuron in neural_network.layers[-1]]
        self.assertEqual(result, sample_data)

    def test_shape(self, shape=(40, 52, 1, 7, 4)):
        factory = MlpFactory()
        neural_creator = factory.make_neural_creator()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        self.assertEqual(neural_network.shape, list(shape))

    def test_single_pattern_forward_action(self):
        pass

    def test_single_pattern_backward_action(self):
        pass

    def test_read_output_layer(self, shape=(4, 6, 78)):
        factory = MlpFactory()
        neural_creator = factory.make_neural_creator()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        sample_data = [random.random() for x in range(shape[-1])]
        for neuron, output_value in zip(neural_network.layers[-1], sample_data):
            neuron.output = output_value
        result = neural_network.read_output_layer()
        self.assertEqual(result, sample_data)


class TestMlpFactory(unittest.TestCase):
    """ Test for MlpFactory, a simple factory for simple multilayer feed forward networks"""

    def test_make_connection(self):
        """  MlpFactory unit test """
        factory = MlpFactory()
        neuron = factory.make_primitive_neuron()
        connection = factory.make_connection(neuron)
        self.assertTrue(connection.neuron is neuron)

    def test_make_container(self):
        """  MlpFactory unit test """
        factory = MlpFactory()
        container = factory.make_container()
        self.assertTrue(isinstance(container, type(Container())))
        self.assertEqual(len(container), 0)

    def test_make_neuron(self):
        """  MlpFactory unit test """
        factory = MlpFactory()
        neuron = factory.make_primitive_neuron()
        self.assertTrue(isinstance(neuron, type(SimpleNeuron())))


class TestSimpleNeuralCreator(unittest.TestCase):
    """ Unit tests for SimpleNeuralCreator class, the builder of SimpleNeuralNetworks
    """

    def test_create_primitive_layers(self):
        """ TestSimpleNeuralCreator Unit tests """
        factory = MlpFactory()
        neural_network = factory.make_primitive_neural_network()
        SimpleNeuralCreator.create_primitive_layers(factory, neural_network, [3, 5, 2])
        self.assertEqual(len(neural_network.layers), 3)
        self.assertEqual(list(map(len, neural_network.layers)), [3, 5, 2])

    def test_connect_and_initialize_network(self):
        factory = MlpFactory()
        neural_network = factory.make_primitive_neural_network()
        SimpleNeuralCreator.create_primitive_layers(factory, neural_network, [3, 5, 2])
        SimpleNeuralCreator.connect_network_layers(factory, neural_network)
        SimpleNeuralCreator.initialize_network(neural_network)

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
