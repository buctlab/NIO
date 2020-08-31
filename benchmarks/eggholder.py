from numpy import array, sin, sqrt
from benchmarks.benchmark import Benchmark


class Eggholder(Benchmark):
    """dim: 2"""
    def __init__(self, lower=-512, upper=512, dimension=2):
        super(Eggholder, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([[512, 404.2319]]), -959.6406627106155

    @staticmethod
    def eval(sol):
        temp1 = -(sol[1] + 47) * sin(sqrt(abs(sol[1] + (sol[0] / 2) + 47)))
        temp2 = - sol[0] * sin(sqrt(abs(sol[0] - (sol[1] + 47))))

        return temp1 + temp2
