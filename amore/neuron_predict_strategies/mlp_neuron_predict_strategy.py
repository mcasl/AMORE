import functools
import operator
from amore.neuron_predict_strategies.neuron_predict_strategy import NeuronPredictStrategy


class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    def __init__(self, neuron):
        NeuronPredictStrategy.__init__(self, neuron)
        self.output_derivative = 0.0
        self.induced_local_field = 0.0

    def __call__(self, *args, **kwargs):
        inputs_x_weights = map((lambda connection: connection.neuron.output * connection.weight),
                               self.neuron.connections)
        self.induced_local_field = functools.reduce(operator.add, inputs_x_weights, self.neuron.bias)
        self.neuron.output = self.neuron.activation_function(self.induced_local_field)
        self.output_derivative = self.neuron.activation_function.derivative(self.induced_local_field)
        return self.neuron.output
