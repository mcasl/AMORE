from neuron cimport Neuron


cdef class MlpConnection:
    cdef public int weight
    cdef readonly Neuron neuron