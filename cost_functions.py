import numpy


############################################################


def __lms(prediction, target):
    residual = prediction - target
    return numpy.mean(residual ** 2)


def __lms_derivative(prediction, target):
    residual = prediction - target
    return residual


__lms.derivative = __lms_derivative


############################################################


def __lmls(prediction, target):
    residual = prediction - target
    result = numpy.mean(numpy.log(1 + residual ** 2 / 2))
    return result


def __lmls_derivative(prediction, target):
    residual = prediction - target
    result = residual / (1 + residual ** 2 / 2)
    return result


__lmls.derivative = __lmls_derivative

############################################################

cost_functions_set = {'LMS': __lms,
                      'LMLS': __lmls
                      }
