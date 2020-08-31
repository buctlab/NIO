from numpy import array, zeros
from benchmarks.benchmark import Benchmark


class Sphere(Benchmark):
    """dim: n"""
    def __init__(self, lower=-5.12, upper=5.12, dimension=2):
        super(Sphere, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(sol):
        val = 0
        for i in range(len(sol)):
            val += sol[i] ** 2
        return val
