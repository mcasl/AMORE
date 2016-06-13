from common cimport *
from materials cimport *
from cost_functions cimport *

cdef class NeuronFitStrategy(object):
    cdef public:
        Neuron neuron
        CostFunction cost_function  # TODO: change object declaration

cdef class MlpNeuronFitStrategy(NeuronFitStrategy):
    pass


cdef class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    cdef public:
        double delta
        double learning_rate
        double output_derivative
        double target

cdef class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    pass

cdef class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    pass
