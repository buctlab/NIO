from numpy import array, zeros, cos, pi, linspace, meshgrid
from benchmarks.benchmark import Benchmark


class Rastrigin(Benchmark):
    """dim: n"""
    def __init__(self, lower=-5.12, upper=5.12, dimension=2):
        super(Rastrigin, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(sol):
        val = 0.0
        for x in sol:
            val = val + x ** 2 - 10 * cos(2 * pi * x)
        
        return 10.0 * len(sol) + val