from numpy import array, sin, exp, sqrt, pi, cos
from benchmarks.benchmark import Benchmark


class MyFakeFunctioin(Benchmark):
    """dim: 2"""

    def __init__(self, lower=-10, upper=10, dimension=2):
        super().__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[1.3491, -1.3491], [1.3491, 1.3491], [-1.3491, 1.3491], [-1.3491, -1.3491]]), -2.0626118504479614

    @staticmethod
    def eval(sol):
        sum = 0
        for i in range(len(sol)):
            sum += sol[i] ** 2
        return sum


if __name__ == '__main__':
    a = MyFakeFunctioin()
    a.plot(scale=0.2)
