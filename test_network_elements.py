import unittest

from hypothesis import given, strategies as st

from amore import *
from container import *
from neuron_predict_strategies import *


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


class TestSimpleNeuralNetwork(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        self.assertEqual(neural_network.layers, factory.make_primitive_container())

    def test_call(self):
        factory = AdaptiveGradientDescentFactory()
        builder = factory.make_neural_network_builder()
        neural_network = builder.create_neural_network(factory, [3, 6, 2], 'tanh', 'identity')
        input_data = numpy.ones((100, 3))
        self.assertTrue((neural_network(input_data) == neural_network.predict_strategy(input_data)).all)

    def test_fit(self):
        pass  # TODO: test

    def test_predict(self):
        pass  # TODO: test

    def test_insert_input_data(self, shape=(40, 3, 2)):
        factory = AdaptiveGradientDescentFactory()
        neural_network = factory.make_primitive_neural_network()
        neural_network_builder = factory.make_neural_network_builder()
        neural_network = neural_network_builder.create_neural_network(factory, shape, 'Tanh', 'Identity')
        sample_data = [random.random() for dummy in range(shape[0])]
        neural_network.read_input_data(sample_data)
        result = numpy.asarray([neuron.output for neuron in neural_network.layers[0]])
        self.assertTrue((result == sample_data).all)

    def test_write_targets_in_output_layer(self, shape=(4, 6, 78)):
        factory = AdaptiveGradientDescentFactory()
        neural_creator = factory.make_neural_network_builder()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        sample_data = [random.random() for dummy in range(shape[-1])]
        neural_network.fit_strategy.write_targets_in_output_layer(sample_data)
        result = [neuron.target for neuron in neural_network.layers[-1]]
        self.assertEqual(result, sample_data)

    def test_shape(self, shape=(40, 52, 1, 7, 4)):
        factory = AdaptiveGradientDescentFactory()
        neural_creator = factory.make_neural_network_builder()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        self.assertEqual(neural_network.shape, list(shape))

    def test_single_pattern_forward_action(self):
        pass

    def test_single_pattern_backward_action(self):
        pass

    def test_read_output_layer(self, shape=(4, 6, 78)):
        factory = AdaptiveGradientDescentFactory()
        neural_creator = factory.make_neural_network_builder()
        neural_network = neural_creator.create_neural_network(factory, shape, 'Tanh', 'Identity')
        sample_data = [random.random() for dummy in range(shape[-1])]
        for neuron, output_value in zip(neural_network.layers[-1], sample_data):
            neuron.output = output_value
        result = neural_network.inspect_output()
        self.assertEqual(result, sample_data)


if __name__ == '__main__':
    unittest.main()
