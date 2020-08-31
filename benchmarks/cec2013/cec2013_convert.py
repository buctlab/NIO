from numpy import zeros
from benchmarks.benchmark import Benchmark
from benchmarks.cec2013.cec2013.cec2013 import CEC2013


class CEC2013Convert(Benchmark):

    def __init__(self, i):
        self.f = CEC2013(i)
        dimension = self.f.get_dimension()
        min_values, max_values = [0] * dimension, [0] * dimension
        for k in range(dimension):
            min_values[k] = self.f.get_lbound(k)
            max_values[k] = self.f.get_ubound(k)

        super(CEC2013Convert, self).__init__(min_values, max_values, dimension)

    def get_optimum(self):
        # [[], [], ..., []], min_value
        # print(__functions_[1])
        return self.f.get_no_goptima(), self.f.get_fitness_goptima()

    def eval(self, sol):
        return self.f.evaluate(sol)

    def evaluate(self, sol):
        return self.eval(sol)

    def get_rho(self):
        return self.f.get_rho()

    def get_fitness_goptima(self):
        return self.f.get_fitness_goptima()

    def get_no_goptima(self):
        return self.f.get_no_goptima()


if __name__ == '__main__':
    for i in range(1, 21):
        f = CEC2013Convert(i)
        # f.plot(scale=0.32)
        no_optimum, goptima = f.get_optimum()
        print("f", i, ":", no_optimum, goptima)
        arr = [1] * f.dimension
        print("val=", f.eval(arr))
