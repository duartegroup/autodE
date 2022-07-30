import numpy as np


def f(x, y):
    return x**2 - y**2


def g(x, y):
    return np.array([2 * x, - 2 * y])


if __name__ == '__main__':

    s = np.array([0.1, 0.1])

    for _ in range(50):

        _g = 0.4 * g(*s)
        s[0] -= _g[0]
        s[1] += _g[1]

        print(s)
