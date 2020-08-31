from numpy import array, ones
from benchmarks.benchmark import Benchmark


class Rosenbrock(Benchmark):
    """dim: n  it may be restricted to the hypercube [-2.048, 2.048]"""
    def __init__(self, lower=-5, upper=10, dimension=2):
        super(Rosenbrock, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([ones(self.dimension)]), 0

    @staticmethod
    def eval(sol):
        d = len(sol)
        val = 0.0
        for i in range(d - 1):
            val += 100.0 * (sol[i + 1] - sol[i] ** 2) ** 2 + (sol[i] - 1) ** 2

        return val
