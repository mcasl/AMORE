from common cimport RealNumber
from cost_functions cimport *
from materials cimport Neuron

cdef class NeuronFitStrategy(object):
    cdef public:
        Neuron neuron
        CostFunction cost_function

cdef class MlpNeuronFitStrategy(NeuronFitStrategy):
    pass

cdef class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    cdef public:
        RealNumber delta
        RealNumber learning_rate
        RealNumber output_derivative
        RealNumber target

cdef class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    pass

cdef class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    pass
