import numpy as np


class CostFunction:
    def __init__(self, cost_function, cost_function_derivative):
        self.original = cost_function
        self.derivative = cost_function_derivative

    def __call__(self, prediction, target):
        return self.original(prediction, target)


############################################################
def adapt_lms_original(prediction, target):
    residual = prediction - target
    return residual ** 2


def adapt_lms_derivative(prediction, target):
    residual = prediction - target
    return residual


############################################################
############################################################


def batch_lms_original(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual ** 2)


def batch_lms_derivative(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual)


############################################################


def adapt_lmls_original(prediction, target):
    residual = prediction - target
    result = np.mean(np.log(1 + residual ** 2 / 2))
    return result


def adapt_lmls_derivative(prediction, target):
    residual = prediction - target
    result = residual / (1 + residual ** 2 / 2)
    return result


############################################################

cost_functions_set = {'adaptLMS': CostFunction(adapt_lms_original, adapt_lms_derivative),
                      'adaptLMLS': CostFunction(adapt_lmls_original, adapt_lmls_derivative),
                      'default': CostFunction(adapt_lms_original, adapt_lms_derivative)
                      }
