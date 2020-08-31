from numpy import array, zeros
from benchmarks.benchmark import Benchmark


class Camel3(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-5, upper=5, dimension=2):
        super(Camel3, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[0.0, 0.0]]), 0.0

    @staticmethod
    def eval(sol):
        term1 = 2 * sol[0] ** 2
        term2 = -1.05 * sol[0] ** 4
        term3 = sol[0] ** 6 / 6
        term4 = sol[0] * sol[1]
        term5 = sol[1] ** 2
        
        return term1 + term2 + term3 + term4 + term5
