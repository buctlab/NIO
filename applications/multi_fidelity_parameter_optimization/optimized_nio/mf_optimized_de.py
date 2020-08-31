from applications.multi_fidelity_parameter_optimization.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import DifferentialEvolution
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedDEFunc')
logger.setLevel('INFO')


class MFOptimizedDEFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(0, 0), upper=(2, 1), benchmark=None, fidelity_option='ReLU', fixed_fidelity=10, scale=10):
        dimension = 2
        super(MFOptimizedDEFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option, fixed_fidelity, scale)

    def get_optimum(self):
        return array([[2, 0.9]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale
        de = DifferentialEvolution(F=params[0], CR=params[1], func=self.benchmark, iterations=iterations)
        best = de.run_return_best_val()
        self.eval_count += de.eval_count
        return best
