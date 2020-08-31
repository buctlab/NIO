from numpy import array, cos
from benchmarks.benchmark import Benchmark


class Shubert(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-10, upper=10, dimension=2):
        super(Shubert, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[]]), -186.7309  # has 18 global minima. May be restricted to the square xi /in [-5.12, 5.12]

    @staticmethod
    def eval(sol):
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            new1 = i * cos((i + 1) * sol[0] + i)
            new2 = i * cos((i + 1) * sol[1] + i)
            sum1 += new1
            sum2 += new2

        return sum1 * sum2
