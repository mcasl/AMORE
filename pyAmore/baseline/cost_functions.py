import numpy as np


class CostFunction:
    def __init__(self, cost_function, cost_function_derivative):
        self.cost_function = cost_function
        self.derivative = cost_function_derivative

    def __call__(self, prediction, target):
        return self.cost_function(prediction, target)


############################################################
def __adapt_lms(prediction, target):
    residual = prediction - target
    return residual ** 2


def __adapt_lms_derivative(prediction, target):
    residual = prediction - target
    return residual

############################################################
############################################################

def __batch_lms(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual ** 2)

def __batch_lms_derivative(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual)
############################################################
def __adapt_lmls(prediction, target):
    residual = prediction - target
    result = np.mean(np.log(1 + residual ** 2 / 2))
    return result


def __adapt_lmls_derivative(prediction, target):
    residual = prediction - target
    result = residual / (1 + residual ** 2 / 2)
    return result
############################################################

cost_functions_set = {'adaptLMS': CostFunction(__adapt_lms, __adapt_lms_derivative),
                      'adaptLMLS': CostFunction(__adapt_lmls, __adapt_lmls_derivative),
                      'default': CostFunction(__adapt_lms, __adapt_lms_derivative)
                      }
