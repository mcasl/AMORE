import unittest

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

    def test_size(self, size=(2, 2, 2)):
        factory = MlpFactory()
        neural_creator = factory.make_neural_creator()
        neural_network = neural_creator.create_neural_network(factory, size, 'Tanh', 'Identity')
        self.assertEqual(neural_network.size(), list(size))


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
