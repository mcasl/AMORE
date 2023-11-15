# cython: profile=True
# cython: linetrace=True

from pyAmore.cython.cost_functions import cost_functions_set
from cost_functions cimport *
from materials cimport *

cdef class NeuronFitStrategy:
    def __init__(self):
        self.cost_function = cost_functions_set['default']


cdef class MlpNeuronFitStrategy(NeuronFitStrategy):
    def __init__(self, neuron):
        NeuronFitStrategy.__init__(self)
        self.neuron = neuron


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
        cdef RealNumber minus_learning_rate_x_delta = -self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        cdef int connection_position
        cdef int number_of_connections = len(self.neuron.connections)
        cdef RealNumber input_value
        cdef MlpConnection connection
        cdef AdaptiveGradientDescentNeuronFitStrategy fit_strategy
        for connection_position in range(number_of_connections):
            connection = neuron.connections[connection_position]
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            fit_strategy = connection.neuron.fit_strategy
            fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0


cdef class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, MlpNeuron neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(self, neuron)

    cpdef fit(self):
        neuron = self.neuron
        self.output_derivative = neuron.activation_function.derivative(neuron.predict_strategy.induced_local_field,
                                                                       neuron.output)
        self.delta *= self.output_derivative
        cdef RealNumber minus_learning_rate_x_delta = - self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        cdef int connection_position
        cdef int number_of_connections = len(self.neuron.connections)
        cdef RealNumber input_value
        cdef MlpConnection connection
        cdef AdaptiveGradientDescentNeuronFitStrategy fit_strategy
        for connection_position in range(number_of_connections):
            connection = neuron.connections[connection_position]
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            fit_strategy = connection.neuron.fit_strategy
            fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0
