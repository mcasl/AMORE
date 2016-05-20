from .mlp_network_predict_strategy import MlpNetworkPredictStrategy


class BatchGradientDescentWithMomentumNetworkPredictStrategy(MlpNetworkPredictStrategy):
    def __init__(self, neural_network):
        MlpNetworkPredictStrategy.__init__(self, neural_network)
