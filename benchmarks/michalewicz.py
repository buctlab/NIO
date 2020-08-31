from numpy import array, pi, sin
from benchmarks.benchmark import Benchmark


class Michalewicz(Benchmark):
    """dim: [2, 5, 10]"""
    def __init__(self, lower=0, upper=pi, dimension=2):
        super(Michalewicz, self).__init__(lower, upper, dimension)
        if self.dimension not in [2, 5, 10]:
            raise ValueError("Michalewicz Function support only [2, 5, 10] dimension.")

    def get_optimum(self):
        if self.dimension == 2:
            return array([[2.20, 1.57]]), -1.8013034101
        elif self.dimension == 5:
            return array([[]]), -4.687658
        elif self.dimension == 10:
            return array([[]]), -9.66015
        else:
            return array([[2.20, 1.57]]), -1.8013034101

    @staticmethod
    def eval(sol):
        val = 0.0
        m = 10
        for i in range(len(sol)):
            val += sin(sol[i]) * sin(((i + 1) * (sol[i] ** 2)) / pi) ** (2 * m)

        return -val
