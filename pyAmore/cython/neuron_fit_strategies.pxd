from common cimport RealNumber
#from cost_functions cimport *
from materials cimport Neuron

cdef class NeuronFitStrategy(object):
    cdef public:
        Neuron neuron
        object cost_function  # TODO: change object declaration

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
