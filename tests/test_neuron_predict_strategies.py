import math
import unittest

from amore.interface import *
from amore.neuron_predict_strategies import NeuronPredictStrategy


class TestNeuronPredictStrategies(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = factory.make_primitive_neuron(neural_network)
        predict_strategy = factory.make_neuron_predict_strategy(neuron)
        self.assertEqual(predict_strategy.neuron, neuron)

    def test_call(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = factory.make_primitive_neuron(neural_network)
        predict_strategy = factory.make_neuron_predict_strategy(neuron)
        self.assertRaises(NotImplementedError, NeuronPredictStrategy.__call__, predict_strategy)


class TestMlpNeuronPredictStrategy(unittest.TestCase):
    def test_init(self):
        factory = AdaptiveGradientDescentMaterialsFactory()
        neural_network = factory.make_primitive_network()
        neuron = factory.make_primitive_neuron(neural_network)
        predict_strategy = factory.make_neuron_predict_strategy(neuron)
        self.assertEqual(predict_strategy.induced_local_field, 0.0)

    def test_call(self):
        neural_network = mlp_network([3, 2, 1], 'tanh', 'tanh')
        neuron = neural_network.layers[2][0]
        accumulator = neuron.bias
        for connection in neuron.connections:
            accumulator += connection.neuron.output * connection.weight
        result = math.tanh(accumulator)
        self.assertEqual(neuron.predict_strategy(), result)
