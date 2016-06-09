from abc import ABCMeta, abstractmethod
import numpy as np
from .activation_functions import activation_functions_set
from .cost_functions import cost_functions_set

cdef class Network(object, metaclass=ABCMeta):
  cdef NeuralFactory factory
  cdef PredictStrategy predict_strategy
  cdef FitStrategy fit_strategy


cdef class MlpNetwork(Network):
    """ Simple implementation of a multilayer feed forward network
    """
    cdef Container layers


cdef class MlpContainer(list)


cdef class Neuron:
    cdef str label
    cdef double output
    cdef NeuralNetwork neural_network
    cdef PredictStrategy predict_strategy
    cdef FitStrategy fit_strategy


class MlpNeuron(Neuron):
    cdef ActivationFunction activation_function
    cdef Container connections
    cdef double bias

class MlpConnection(object):
    cdef double weight
    cdef Neuron neuron
