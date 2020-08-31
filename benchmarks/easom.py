from numpy import pi, cos, exp, array
from benchmarks.benchmark import Benchmark


class Easom(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-100, upper=100, dimension=2):
        super(Easom, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[pi, pi]]), -1

    @staticmethod
    def eval(sol):
        term1 = -cos(sol[0]) * cos(sol[1])
        term2 = exp(-(sol[0] - pi) ** 2 - (sol[1] - pi) ** 2)
        
        return term1 * term2
