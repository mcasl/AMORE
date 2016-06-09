from Neuron cimport Neuron



cdef class MlpConnection:

    def __init__(self, neuron, weight=0.0):
        self.weight = weight
        self.neuron = neuron
