from .neuralnetwork import NeuralNetwork


class MlpNeuralNetwork(NeuralNetwork):
    """ Simple implementation of a multilayer feed forward network
    """

    def __init__(self, neural_factory):
        NeuralNetwork.__init__(self, neural_factory)
        self.layers = neural_factory.make_primitive_container()

    # TODO:  cost_function = neural_factory.make_cost_function('LMS')

    def __call__(self, input_data):
        return self.predict_strategy(input_data)

    def read(self, input_data):
        for neuron, value in zip(self.layers[0], input_data):
            neuron.output = value

    def inspect_output(self):
        return [neuron.output for neuron in self.layers[-1]]

    @property
    def shape(self):
        """ Gives information on the number of neurons in the neural network
        """
        return list(map(len, self.layers))
