from numpy import array, sqrt
from benchmarks.benchmark import Benchmark


class Bukin6(Benchmark):
    """dim: 2"""
    def __init__(self, lower=(-15, -3), upper=(-5, 3), dimension=2):
        super(Bukin6, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[-10, 1]]), 0.0

    @staticmethod
    def eval(sol):
        x1 = sol[0]
        x2 = sol[1]
        term1 = 100 * sqrt(abs(x2 - 0.01 * x1 ** 2))
        term2 = 0.01 * abs(x1 + 10)
        
        return term1 + term2


if __name__ == '__main__':
    bukin6 = Bukin6(lower=(-15, -3), upper=(-5, 3), dimension=2)
    print(bukin6.get_optimum())
    bukin6.plot(scale=0.32)
