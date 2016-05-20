from .mlp_neuron_fit_strategy import MlpNeuronFitStrategy


class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        MlpNeuronFitStrategy.__init__(self, neuron)
        self.delta = 0.0
        self.learning_rate = 0.1
        self.output_derivative = 0.0
        self.target = 0.0
