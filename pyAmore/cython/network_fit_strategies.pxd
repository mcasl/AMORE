from .neuron_predict_strategies import *


cdef class NetworkFitStrategy(object, metaclass=ABCMeta):
    cdef NeuralNetwork neural_network


class MlpNetworkFitStrategy(NetworkFitStrategy):
    cdef Container neuron_fit_sequence


cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
