import amore.interface as am
import numpy as np


def test():
    data = np.random.rand(1000, 1)
    target = data ** 2
    net = am.mlp_network([1, 5, 1], 'tanh', 'identity')
    print('\nNet created.\n')
    am.fit_adaptive_gradient_descent(net, data, target, 0.1, 10, 5)
    print('\nFinished.\n')


if __name__ == '__main__':
    import timeit

    timer = timeit.Timer("test()", setup="from __main__ import test").repeat(repeat=3, number=1)
    print(min(timer))
    # %timeit amore.interface.fit_adaptive_gradient_descent(net, data, target, 0.1, 10, 5)
    # 1 loops, best of 3: 0.883 s per loop
