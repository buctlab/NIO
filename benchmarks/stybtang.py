from numpy import array, full
from benchmarks.benchmark import Benchmark


class Stybtang(Benchmark):
    """dim: n"""
    def __init__(self, lower=-5, upper=5, dimension=2):
        super(Stybtang, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([full(self.dimension, -2.903534)]), -39.16599 * self.dimension

    @staticmethod
    def eval(sol):
        d = len(sol)
        val = 0.0
        for i in range(d):
            xi = sol[i]
            new = xi ** 4 - 16 * xi ** 2 + 5 * xi
            val += new

        return val / 2
