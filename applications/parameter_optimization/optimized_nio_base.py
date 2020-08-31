from numpy import ndarray, zeros, array, full
import logging

logging.basicConfig()
logger = logging.getLogger('OptimizedNIOBase')
logger.setLevel('INFO')


class OptimizedNIOBase:

    def __init__(self, lower, upper, dimension, benchmark):
        self.dimension = dimension
        if isinstance(lower, (float, int)):
            self.lower = full(self.dimension, lower)
            self.upper = full(self.dimension, upper)
        elif isinstance(lower, (ndarray, list, tuple)) and len(lower) == dimension:
            self.lower = array(lower)
            self.upper = array(upper)
        else:
            raise ValueError("{bench}: Type mismatch or Length of bound mismatch with dimension".format(bench=self.__class__.__name__))

        self.benchmark = benchmark
        self.eval_count = 0

    def get_optimum(self):
        return array([zeros(self.dimension)]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        pass
