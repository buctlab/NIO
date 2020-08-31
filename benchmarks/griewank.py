from numpy import array, zeros, cos, sqrt
from benchmarks.benchmark import Benchmark


class Griewank(Benchmark):
    """dim: n"""
    def __init__(self, lower=-600, upper=600, dimension=2):
        super(Griewank, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(sol):
        d = len(sol)
        sum = 0.0
        prod = 1

        for i in range(d):
            xi = sol[i]
            sum += xi ** 2 / 4000
            prod *= cos(xi / sqrt(i + 1))

        return sum - prod + 1
