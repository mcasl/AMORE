import unittest
from amore import *


class TestMlpPredictStrategy(unittest.TestCase):
    """ Unit tests for MLpPredictStrategy
    """

    def test__call__(self):
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

        # Expected result calculation
        accumulator = test_neuron.bias
        for connection in test_neuron.connections:
            accumulator += connection.neuron.output * connection.weight
        expected_result = test_neuron.activation_function(accumulator)
        self.assertEqual(test_neuron.predict_strategy(), expected_result)


if __name__ == '__main__':
    unittest.main()
