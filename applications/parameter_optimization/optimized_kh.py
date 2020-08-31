from applications.parameter_optimization.optimized_nio_base import OptimizedNIOBase
from algorithms import KrillHerd
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedKHFunc')
logger.setLevel('INFO')


class OptimizedKHFunc(OptimizedNIOBase):

    def __init__(self, lower=(0, 0), upper=(0.1, 0.1), dimension=2, benchmark=None):
        super(OptimizedKHFunc, self).__init__(lower, upper, dimension, benchmark)

    def get_optimum(self):
        return array([[0.01, 0.02]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        kh = KrillHerd(N_max=params[0], V_f=params[1], func=self.benchmark, iterations=200)
        best = kh.run_return_best_val()
        self.eval_count += kh.eval_count
        return best
