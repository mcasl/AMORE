import numpy as np

from .neuron_predict_strategies import *


cdef class NetworkPredictStrategy(object, metaclass=ABCMeta):
    cdef NeuralNetwork neural_network


cdef class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    cdef NeuronPredictStrategy neuron_predict_strategy

