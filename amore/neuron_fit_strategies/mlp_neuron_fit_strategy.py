from abc import abstractmethod
from amore.neuron_fit_strategies.neuron_fit_strategy import NeuronFitStrategy


class MlpNeuronFitStrategy(NeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        NeuronFitStrategy.__init__(self, neuron)
