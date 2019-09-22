from math import pi, sin, cos
import random


def f1(z):
    return sin(z)


def f2(z):
    return cos(z)


def f3(z):
    return z


def f(z):
    return -pi + 0.565*f1(z) + 2.657*f2(z) + 0.674*f3(z)


def pre_processing(size=15):
    step = (2*pi)/size

    X = []
    y = []

    n = 0
    while n <= 2*pi:
        X.append([f1(n), f2(n), f3(n)])
        y.append(f(n))
        n += step

    return X, y


def generate_random_cases(size=15):
    X = []
    y = []

    for i in range(size):
        n = random.uniform(0, 2*pi)
        X.append([f1(n), f2(n), f3(n)])
        y.append(f(n))

    return X, y
