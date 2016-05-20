from .mlp_factory import MlpFactory

class BatchGradientDescentWithMomentumFactory(MlpFactory):
    @staticmethod
    def make_neuron_fit_strategy(neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return BatchDescentWithMomentumOutputNeuronFitStrategy(neuron)
        else:
            return BatchDescentWithMomentumHiddenNeuronFitStrategy(neuron)

    @staticmethod
    def make_neural_network_predict_strategy(neural_network):
        return BatchDescentWithMomentumNetworkPredictStrategy(neural_network)
