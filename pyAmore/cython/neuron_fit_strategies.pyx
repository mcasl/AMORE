# cython: profile=True

from pyAmore.cython.cost_functions import cost_functions_set
from cost_functions cimport *
from materials cimport *

cdef class NeuronFitStrategy:
    def __init__(self, Neuron neuron):
        self.neuron = neuron
        self.cost_function = cost_functions_set['default']


cdef class MlpNeuronFitStrategy(NeuronFitStrategy):
    def __init__(self, MlpNeuron neuron):
        NeuronFitStrategy.__init__(self, neuron)


cdef class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    def __init__(self, MlpNeuron neuron):
        MlpNeuronFitStrategy.__init__(self, neuron)
        self.delta = 0.0
        self.learning_rate = 0.1
        self.output_derivative = 0.0
        self.target = 0.0


cdef class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, MlpNeuron neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(self, neuron)

    cpdef fit(self):
        neuron = self.neuron
        self.output_derivative = neuron.activation_function.derivative(neuron.predict_strategy.induced_local_field,
                                                                       neuron.output)
        self.delta = self.output_derivative * self.cost_function.derivative(neuron.output, self.target)
        minus_learning_rate_x_delta = -self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        for connection in self.neuron.connections:
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0


cdef class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, MlpNeuron neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(self, neuron)

    cpdef fit(self):
        neuron = self.neuron
        self.output_derivative = neuron.activation_function.derivative(neuron.predict_strategy.induced_local_field,
                                                                       neuron.output)
        self.delta *= self.output_derivative
        minus_learning_rate_x_delta = - self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        cdef int position
        cdef int neuron_connections_length = len(self.neuron.connections)
        for position in range(neuron_connections_length):
            connection = self.neuron.connections[position]
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0
