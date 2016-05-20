from .mlp_factory import MlpFactory


class BatchGradientDescentFactory(MlpFactory):
    def make_neuron_fit_strategy(neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return BatchGradientDescentOutputNeuronFitStrategy(neuron)
        else:
            return BatchGradientDescentHiddenNeuronFitStrategy(neuron)

    def make_neural_network_predict_strategy(self, neural_network):
        return BatchGradientDescentNetworkPredictStrategy(neural_network)
