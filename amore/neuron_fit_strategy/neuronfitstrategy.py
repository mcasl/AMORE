from amore.network_elements.cost_functions import *


class NeuronFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neuron):
        self.neuron = neuron
        self.cost_function = cost_functions_set['LMS']

