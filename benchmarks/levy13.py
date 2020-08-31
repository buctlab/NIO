from numpy import array, pi, sin
from benchmarks.benchmark import Benchmark


class Levy13(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-10, upper=10, dimension=2):
        super(Levy13, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[1, 1]]), 1.3497838043956716e-31

    @staticmethod
    def eval(sol):
        temp1 = (sin(3 * pi * sol[0]) ** 2)
        temp2 = (sol[0] - 1) ** 2 * (1 + (sin(3 * pi * sol[1])) ** 2)
        temp3 = (sol[1] - 1) ** 2 * (1 + (sin(2 * pi * sol[1])) ** 2)

        return temp1 + temp2 + temp3
