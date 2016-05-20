from .mlp_neuron_predict_strategy import MlpNeuronPredictStrategy


class BatchGradientDescentNeuronPredictStrategy(MlpNeuronPredictStrategy):
    def __init__(self, neuron):
        MlpNeuronPredictStrategy.__init__(self, neuron)
