from .mlp_network_fit_strategy import MlpNetworkFitStrategy


class AdaptiveGradientDescentNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    def __call__(self, input_data, target_data):
        for input_data_row, target_data_row in zip(input_data, target_data):
            self.neural_network.read(input_data_row)
            self.neural_network.predict_strategy.activate_neurons()
            self.neural_network.fit_strategy.write_targets_in_output_layer(target_data_row)
            self.neural_network.fit_strategy.single_pattern_backward_action()
