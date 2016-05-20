from .neuronfitstrategy import NeuronFitStrategy


class MlpNeuronFitStrategy(NeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        NeuronFitStrategy.__init__(self, neuron)
