from common cimport RealNumber
from cost_functions cimport *
from materials cimport Neuron

cdef class NeuronFitStrategy:
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
    cpdef fit(self)

cdef class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    cpdef fit(self)
