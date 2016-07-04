from common                     cimport RealNumber
from network_fit_strategies     cimport NetworkFitStrategy
from network_predict_strategies cimport NetworkPredictStrategy
from neuron_fit_strategies      cimport MlpNeuronFitStrategy
from neuron_predict_strategies  cimport MlpNeuronPredictStrategy
from activation_functions       cimport ActivationFunction

cdef class MlpContainer(list):
    pass

cdef class Network:
    cdef public:
        object factory
        NetworkPredictStrategy predict_strategy
        NetworkFitStrategy fit_strategy
    cpdef poke_inputs(self, input_data)
    cpdef pick_outputs(self)

cdef class MlpNetwork(Network):
    cdef public:
        MlpContainer layers
    cpdef poke_inputs(self, input_data)
    cpdef pick_outputs(self)

cdef class Neuron:
    cdef public:
        str label
        RealNumber output


cdef class MlpNeuron(Neuron):
    cdef public:
        ActivationFunction activation_function
        MlpContainer connections
        RealNumber bias
        MlpNetwork neural_network
        MlpNeuronPredictStrategy predict_strategy
        MlpNeuronFitStrategy fit_strategy

cdef class MlpConnection:
    cdef public:
        RealNumber weight
        MlpNeuron neuron
