from applications.parameter_optimization.optimized_nio_base import OptimizedNIOBase
from algorithms import WaterWaveOptimization
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedWWOFunc')
logger.setLevel('INFO')


class OptimizedWWOFunc(OptimizedNIOBase):

    def __init__(self, lower=(1.001, 0), upper=(1.01, 1), dimension=2, benchmark=None):
        super(OptimizedWWOFunc, self).__init__(lower, upper, dimension, benchmark)

    def get_optimum(self):
        return array([[1.0026, 0.5]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        wwo = WaterWaveOptimization(alpha=params[0], lamb=params[1], func=self.benchmark, iterations=200)
        best = wwo.run_return_best_val()
        self.eval_count += wwo.eval_count
        return best
