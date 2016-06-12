from .cost_functions import *
from .neuron_predict_strategies import *


cdef class NeuronFitStrategy(object, metaclass=ABCMeta):
    cdef Neuron neuron
    cdef CostFunction cost_function


cdef class MlpNeuronFitStrategy(NeuronFitStrategy)


cdef class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    cdef double delta
    cdef double learning_rate
    cdef double output_derivative
    cdef double target


cdef class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy)


cdef class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy)
