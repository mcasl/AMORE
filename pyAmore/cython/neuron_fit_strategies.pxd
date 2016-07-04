from common cimport RealNumber
from cost_functions cimport *
from materials cimport MlpNeuron

cdef class NeuronFitStrategy:
    cdef public:
        CostFunction cost_function

cdef class MlpNeuronFitStrategy(NeuronFitStrategy):
    cdef public:
        MlpNeuron neuron

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
