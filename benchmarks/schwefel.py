from numpy import array, full, sqrt, sin, abs
from benchmarks.benchmark import Benchmark


class Schwefel(Benchmark):
    """dim: n"""
    def __init__(self, lower=-500, upper=500, dimension=2):
        super(Schwefel, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([full(self.dimension, 420.9687)]), 2.545567497236334e-05

    @staticmethod
    def eval(sol):
        val = 0
        for x in sol:
            val = val + x * sin(sqrt(abs(x)))
        
        return 418.9829 * len(sol) - val
