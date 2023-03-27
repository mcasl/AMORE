import numpy as np


class CostFunction:
    def __init__(self):
        self.label = None

    def original(self, prediction, target):
        residual = prediction - target
        return residual ** 2

    def derivative(self, prediction, target):
        return prediction - target


class AdaptLmsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'AdaptLms'

    def original(self, prediction, target):
        residual = prediction - target
        return residual ** 2

    def derivative(self, prediction, target):
        return prediction - target


class AdaptLmLsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'AdaptLmLs'

    def original(self, prediction, target):
        residual = prediction - target
        return np.mean(np.log(1 + residual ** 2 / 2))

    def derivative(self, prediction, target):
        residual = prediction - target
        return residual / (1 + residual ** 2 / 2)


class BatchLmsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'BatchLms'

    def original(self, prediction, target):
        residual = prediction - target
        return np.mean(residual ** 2)


    def derivative(self, prediction, target):
        residual = prediction - target
        return np.mean(residual)


############################################################

cost_functions_set = {'adaptLMS': AdaptLmsCostFunction(),
                      'adaptLMLS': AdaptLmLsCostFunction(),
                      'default': CostFunction(),
                      }
