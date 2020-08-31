from numpy import array
from benchmarks.benchmark import Benchmark


class Camel6(Benchmark):
    """dim: 2"""
    def __init__(self, lower=(-3, -2), upper=(3, 2), dimension=2):
        super(Camel6, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[0.0898, -0.7126], [-0.0898, 0.7126]]), -1.0316284229280819

    @staticmethod
    def eval(sol):
        term1 = (4 - 2.1 * sol[0] ** 2 + (sol[0] ** 4) / 3) * sol[0] ** 2
        term2 = sol[0] * sol[1]
        term3 = (-4 + 4 * sol[1] ** 2) * sol[1] ** 2

        return term1 + term2 + term3
