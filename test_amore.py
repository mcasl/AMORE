import unittest

from hypothesis import given, strategies as st

from amore import Connection, SimpleNeuron, MlpFactory, SimpleNeuralCreator
from container import Container


class TestConnection(unittest.TestCase):
    """ Tests for Connection class """

    @given(label=st.integers(), weight=st.floats())
    def test_init(self, label, weight):
        """ Connection unit tests  """
        my_neuron = SimpleNeuron()
        my_neuron.label = label
        my_connection = Connection(my_neuron, weight)
        self.assertTrue(my_connection.neuron.label is label)
        self.assertTrue(my_connection.weight is weight)


class TestSimpleNeuron(unittest.TestCase):
    """ Tests for SimpleNeuron class, a simple multilayer feed forward
        neural network neuron"""
    @given(label_list=st.lists(st.integers()))
    def test_add_connection(self, label_list):
        """  SimpleNeuron unit tests """
        # Setup test
        my_neuron = SimpleNeuron()
        for label in label_list:
            target_neuron = SimpleNeuron()
            target_neuron.label = label
            connection = Connection(neuron=target_neuron, weight=0.0)
            my_neuron.connections.append(connection)
        # Test action
        appended_labels = [connection.neuron.label for connection in my_neuron.connections]
        self.assertEquals(label_list, appended_labels)


class TestMlpFactory(unittest.TestCase):
    """ Test for MlpFactory, a simple factory for simple multilayer feed forward networks"""
    def test_make_connection(self):
        """  MlpFactory unit test """
        factory = MlpFactory()
        neuron = factory.make_neuron(1)
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
        neuron = factory.make_neuron(1)
        self.assertTrue(isinstance(neuron, type(SimpleNeuron())))
        self.assertEqual(neuron.label, 1)

    def test_make_neuron__with_neuron_sequence_for_building_connections(self):
        """  MlpFactory unit test """
        factory = MlpFactory()
        neural_network = factory.make_neural_network(factory)
        neuron_container = factory.make_container()

        for id_iter in range(10):
            neuron_container.append(factory.make_neuron(id_iter))

        neuron = factory.make_neuron(10, neuron_container, 10, neural_network)
        self.assertTrue(isinstance(neuron, type(SimpleNeuron())))
        self.assertEqual(len(neuron.connections), 10)
        self.assertEqual(10, neuron.label)


class TestSimpleNeuralCreator(unittest.TestCase):
    """ Unit tests for SimpleNeuralCreator class, the builder of SimpleNeuralNetworks
    """
    def test_create_neural_network(self):
        """ TestSimpleNeuralCreator Unit tests """
        factory = MlpFactory()
        neural_network = factory.make_neural_network(factory)
        SimpleNeuralCreator.populate_network(factory, neural_network, [3, 5, 2])
        self.assertEqual(len(neural_network.layers), 3)
        self.assertEqual(list(map(len, neural_network.layers)), [3, 5, 2])


if __name__ == '__main__':
    unittest.main()
