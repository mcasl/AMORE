from abc import ABCMeta, abstractmethod


class Neuron(object, metaclass=ABCMeta):
    """ The mother of all neurons (a.k.a. Interface)
    """

    @abstractmethod
    def __init__(self, neural_network):
        """ Initializer. An assumption is made that all neurons will have at least these properties.
        """
        self.label = None
        self.output = 0.0
        self.neural_network = neural_network
        self.predict_strategy = neural_network.factory.make_neuron_predict_strategy(self)
        # self.fit_strategy should not be assigned here as it will depend on the neurons role
        # and it will be the builder's responsibility to assign it

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling Neuron.__call__()")
