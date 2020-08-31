from applications.parameter_optimization.optimized_nio_base import OptimizedNIOBase
from algorithms import SquirrelSearchAlgorithm
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedSSAFunc')
logger.setLevel('INFO')


class OptimizedSSAFunc(OptimizedNIOBase):

    def __init__(self, lower=(1, 0.1, 16), upper=(10, 0.9, 37), dimension=3, benchmark=None):
        super(OptimizedSSAFunc, self).__init__(lower, upper, dimension, benchmark)

    def get_optimum(self):
        return array([[1.9, 0.1, 18]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        ssa = SquirrelSearchAlgorithm(Gc=params[0], Pdp=params[1], sf=params[2], func=self.benchmark, iterations=200)
        best = ssa.run_return_best_val()
        self.eval_count += ssa.eval_count
        return best
