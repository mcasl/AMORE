from amore.neuron_predict_strategies.mlp_neuron_predict_strategy import MlpNeuronPredictStrategy


class AdaptiveGradientDescentWithMomentumNeuronPredictStrategy(MlpNeuronPredictStrategy):
    def __init__(self, neuron):
        MlpNeuronPredictStrategy.__init__(self, neuron)
