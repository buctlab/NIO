from applications.parameter_optimization.optimized_nio_base import OptimizedNIOBase
from algorithms import DifferentialEvolution
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedDEFunc')
logger.setLevel('INFO')


class OptimizedDEFunc(OptimizedNIOBase):

    def __init__(self, lower=(0, 0), upper=(2, 1), dimension=2, benchmark=None):
        super(OptimizedDEFunc, self).__init__(lower, upper, dimension, benchmark)

    def get_optimum(self):
        return array([[2, 0.9]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        de = DifferentialEvolution(F=params[0], CR=params[1], func=self.benchmark, iterations=200)
        best = de.run_return_best_val()
        self.eval_count += de.eval_count
        return best
