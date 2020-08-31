from numpy import zeros, pi, cos, exp, sqrt, array
from benchmarks.benchmark import Benchmark


class Ackley(Benchmark):
    """dim: n"""

    def __init__(self, lower=-32.768, upper=32.768, dimension=2):
        super(Ackley, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(sol):
        a = 20
        b = 0.2
        c = 2 * pi
        d = len(sol)

        sum1 = 0.0
        sum2 = 0.0
        for i in range(d):
            xi = sol[i]
            sum1 = sum1 + xi ** 2
            sum2 = sum2 + cos(c * xi)
        term1 = -a * exp(-b * sqrt(sum1 / d))
        term2 = -exp(sum2 / d)

        return term1 + term2 + a + exp(1)


if __name__ == '__main__':
    ackley = Ackley()
    print(ackley.get_optimum())
    ackley.plot(scale=0.32)
