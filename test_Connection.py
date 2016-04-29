import unittest
# from hypothesis import given, strategies as st
# import container
from connection import Connection
from neuron import SimpleNeuron


class TestConnection(unittest.TestCase):
    """ Tests for Connection class """

    def test_Connection_init___iterable_is_None(self):
        my_neuron = SimpleNeuron()
        my_neuron.id = 1
        my_connection = Connection(my_neuron, 4.5)
        self.assertEqual(my_connection.id, 1)
        self.assertEqual(my_connection.weight, 4.5)


if __name__ == '__main__':
    unittest.main()
