from applications.multi_fidelity_parameter_optimization.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import CuckooSearch
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedCSFunc')
logger.setLevel('INFO')


class MFOptimizedCSFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(0, 0, 1), upper=(0.5, 2, 2), benchmark=None, fidelity_option='ReLU', fixed_fidelity=10, scale=10):
        dimension = 3
        super(MFOptimizedCSFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option, fixed_fidelity, scale)

    def get_optimum(self):
        return array([[0.25, 1, 1.5]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale
        cs = CuckooSearch(pa=params[0], alpha=params[1], lamb=params[2], func=self.benchmark, iterations=iterations)
        best = cs.run_return_best_val()
        self.eval_count += cs.eval_count
        return best
