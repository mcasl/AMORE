from common                     cimport RealNumber
from network_fit_strategies     cimport NetworkFitStrategy
from network_predict_strategies cimport NetworkPredictStrategy
from neuron_fit_strategies      cimport NeuronFitStrategy
from neuron_predict_strategies  cimport NeuronPredictStrategy
from activation_functions       cimport ActivationFunction

cdef class MlpContainer(list):
    pass

cdef class Network:
    cdef public:
        object factory
        NetworkPredictStrategy predict_strategy
        NetworkFitStrategy fit_strategy

cdef class MlpNetwork(Network):
    cdef public:
        MlpContainer layers

cdef class Neuron:
    cdef public:
        str label
        RealNumber output
        Network neural_network
        NeuronPredictStrategy predict_strategy
        NeuronFitStrategy fit_strategy

cdef class MlpNeuron(Neuron):
    cdef public:
        ActivationFunction activation_function  # TODO: must be change to specific type
        MlpContainer connections
        RealNumber bias

cdef class MlpConnection:
    cdef public:
        RealNumber weight
        Neuron neuron
