import numpy as np


############################################################

def __adapt_lms(prediction: float, target: float):
    residual = prediction - target
    return residual ** 2


def __adapt_lms_derivative(prediction: float, target: float):
    residual = prediction - target
    return residual


__adapt_lms.derivative = __adapt_lms_derivative


############################################################
############################################################

def __batch_lms(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual ** 2)


def __batch_lms_derivative(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual)


__batch_lms.derivative = __batch_lms_derivative


############################################################


def __adapt_lmls(prediction: float, target: float):
    residual = prediction - target
    result = np.mean(np.log(1 + residual ** 2 / 2))
    return result


def __adapt_lmls_derivative(prediction: float, target: float):
    residual = prediction - target
    result = residual / (1 + residual ** 2 / 2)
    return result


__adapt_lmls.derivative = __adapt_lmls_derivative

############################################################

cost_functions_set = {'adaptLMS': __adapt_lms,
                      'adaptLMLS': __adapt_lmls,
                      'batchLMS': __batch_lms,
                      'default': __adapt_lms
                      }
