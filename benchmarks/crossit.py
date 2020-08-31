from numpy import array, sin, exp, sqrt, pi
from benchmarks.benchmark import Benchmark


class Crossit(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-10, upper=10, dimension=2):
        super(Crossit, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[1.3491, -1.3491], [1.3491, 1.3491], [-1.3491, 1.3491], [-1.3491, -1.3491]]), -2.0626118504479614

    @staticmethod
    def eval(sol):
        term1 = sin(sol[0]) * sin(sol[1])
        term2 = exp(abs(100 - sqrt(sol[0] ** 2 + sol[1] ** 2) / pi))
        
        return -0.0001 * (abs(term1 * term2) + 1) ** 0.1
