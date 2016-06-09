cdef class Neuron:
    """ The mother of all neurons (a.k.a. Interface)
    """
    def __init__(self):

    #def __init__(self, neural_network):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.output = 0.0
        #self.neural_network = neural_network
        #self.predict_strategy = None
        #self.fit_strategy = None
        # self.fit_strategy should not be assigned here as it might depend on the neurons role
        # and it will be the builders's responsibility to assign it
        # Similarly, self.predict_strategy is not assigned here for versatility.
        # It's the builder that assigns it.

