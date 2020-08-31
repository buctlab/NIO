from applications.parameter_optimization.optimized_nio_base import OptimizedNIOBase
from algorithms import CuckooSearch
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedCSFunc')
logger.setLevel('INFO')


class OptimizedCSFunc(OptimizedNIOBase):

    def __init__(self, lower=(0, 0, 1), upper=(0.5, 2, 2), dimension=3, benchmark=None):
        super(OptimizedCSFunc, self).__init__(lower, upper, dimension, benchmark)

    def get_optimum(self):
        return array([[0.25, 1, 1.5]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        cs = CuckooSearch(pa=params[0], alpha=params[1], lamb=params[2], func=self.benchmark, iterations=200)
        best = cs.run_return_best_val()
        self.eval_count += cs.eval_count
        return best
