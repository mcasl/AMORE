import numpy as np

import pyAmore.cython.interface as am

"""
Baseline Reference:
    am.fit_adaptive_gradient_descent(net, data, target, 0.1, 10, 5)
2.1131909820001056
Process finished with exit code 0
"""


def test():
    data = np.random.rand(1000, 1)
    target = data ** 2
    net = am.mlp_network([1, 50, 1], 'tanh', 'identity')
    print('\nNet created.\n')
    am.fit_adaptive_gradient_descent(net, data, target, 0.1, 10, 10)
    print('\nFinished.\n')
    return net


if __name__ == '__main__':
    import timeit

    timer = timeit.Timer("test()", setup="from __main__ import test").repeat(repeat=6, number=1)
    print(min(timer))
    # %timeit pyAmore.interface.fit_adaptive_gradient_descent(net, data, target, 0.1, 10, 5)
    # 1 loops, best of 3: 0.883 s per loop
