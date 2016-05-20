from .mlp_factory import MlpFactory


class BatchGradientDescentWithMomentumFactory(MlpFactory):
    def make_neuron_fit_strategy(neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return BatchDescentWithMomentumOutputNeuronFitStrategy(neuron)
        else:
            return BatchDescentWithMomentumHiddenNeuronFitStrategy(neuron)

    def make_neural_network_predict_strategy(self, neural_network):
        return BatchDescentWithMomentumNetworkPredictStrategy(neural_network)
