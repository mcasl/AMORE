from abc import ABCMeta, abstractmethod


class NeuralNetwork(object, metaclass=ABCMeta):
    """ The mother of all neural networks (a.k.a Interface)
    """

    @abstractmethod
    def __init__(self, neural_factory):
        self.factory = neural_factory
        self.predict_strategy = neural_factory.make_neural_network_predict_strategy(self)
        self.fit_strategy = neural_factory.make_neural_network_fit_strategy(self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Method for obtaining outputs from inputs
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.__call__")

    @abstractmethod
    def read(self, input_data):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.read")

    @abstractmethod
    def inspect_output(self):
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.inspect_output")

    @abstractmethod
    def shape(self):
        """ Gives information about the number of neurons in the neural network
        """
        raise NotImplementedError("You shouldn't be calling NeuralNetwork.shape")
