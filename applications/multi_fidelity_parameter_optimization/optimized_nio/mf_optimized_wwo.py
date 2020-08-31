from applications.multi_fidelity_parameter_optimization.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import WaterWaveOptimization
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedWWOFunc')
logger.setLevel('INFO')


class MFOptimizedWWOFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(1.001, 0), upper=(1.01, 1), benchmark=None, fidelity_option='ReLU', fixed_fidelity=10, scale=10):
        dimension = 2
        super(MFOptimizedWWOFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option, fixed_fidelity, scale)

    def get_optimum(self):
        return array([[1.0026, 0.5]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale
        wwo = WaterWaveOptimization(alpha=params[0], lamb=params[1], func=self.benchmark, iterations=iterations)
        best = wwo.run_return_best_val()
        self.eval_count += wwo.eval_count
        return best
