cdef class Neuron:
    """ The mother of all neurons (a.k.a. Interface)
    """
    cdef public:
        double output
        str label
