from numpy import array, sin
from benchmarks.benchmark import Benchmark


class Schaffer2(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-100, upper=100, dimension=2):
        super(Schaffer2, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[0, 0]]), 0.0

    @staticmethod
    def eval(sol):
        part1 = (sin(sol[0] ** 2 - sol[1] ** 2)) ** 2 - 0.5
        part2 = (1 + 0.001 * (sol[0] ** 2 + sol[1] ** 2)) ** 2

        return 0.5 + part1 / part2
