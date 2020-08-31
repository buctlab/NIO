from applications.parameter_optimization.optimized_nio_base import OptimizedNIOBase
from algorithms import ParticleSwarmOptimization
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedPSOFunc')
logger.setLevel('INFO')


class OptimizedPSOFunc(OptimizedNIOBase):

    def __init__(self, lower=(0, 0, 0, 0), upper=(1, 10, 10, 20), dimension=4, benchmark=None):
        super(OptimizedPSOFunc, self).__init__(lower, upper, dimension, benchmark)

    def get_optimum(self):
        return array([[0.7, 2.0, 2.0, 4.0]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        pso = ParticleSwarmOptimization(w=params[0], c1=params[1], c2=params[2], v_max=params[3], func=self.benchmark, iterations=200)
        best = pso.run_return_best_val()
        self.eval_count += pso.eval_count
        return best
