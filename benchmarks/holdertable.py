from numpy import array, sin, cos, exp, sqrt, pi
from benchmarks.benchmark import Benchmark


class Holdertable(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-10, upper=10, dimension=2):
        super(Holdertable, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, 9.66459], [-8.05502, -9.66459]]), -19.208502567767606

    @staticmethod
    def eval(sol):
        term1 = sin(sol[0]) * cos(sol[1])
        term2 = exp(abs(1 - sqrt(sol[0] ** 2 + sol[1] ** 2) / pi))
        
        return -abs(term1 * term2)
