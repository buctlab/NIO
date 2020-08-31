from applications.multi_fidelity_parameter_optimization.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import KrillHerd
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedKHFunc')
logger.setLevel('INFO')


class MFOptimizedKHFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(0, 0), upper=(0.1, 0.1), benchmark=None, fidelity_option='ReLU', fixed_fidelity=10, scale=10):
        dimension = 2
        super(MFOptimizedKHFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option, fixed_fidelity, scale)

    def get_optimum(self):
        return array([[0.01, 0.02]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale
        kh = KrillHerd(N_max=params[0], V_f=params[1], func=self.benchmark, iterations=iterations)
        best = kh.run_return_best_val()
        self.eval_count += kh.eval_count
        return best
