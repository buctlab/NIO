from applications.parameter_optimization import OptimizedNIOBase
from numpy import array, zeros, ceil, exp, sin, power, pi
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedNIOBase')
logger.setLevel('INFO')


class MFOptimizedNIOBase(OptimizedNIOBase):
    """Multi-Fidelity Optimized NIO"""
    def __init__(self, lower, upper, dimension, benchmark, fidelity_option='ReLU', fixed_fidelity=10, scale=10):
        super().__init__(lower, upper, dimension, benchmark)
        self.FidelityStrategy = {
            'Fixed': self.fidelity_fixed,
            'ReLU': self.fidelity_relu,
            'Sigmoid': self.fidelity_sigmoid,
            'Sin': self.fidelity_sin,
            'Power': self.fidelity_power,
        }

        self.fidelity = 1
        self.max_fidelity = 10
        if fixed_fidelity > self.max_fidelity:
            raise ValueError("{func}: Fidelity value should be an integer in [1, {max_f}]".format(
                func=self.__class__.__name__, max_f=self.max_fidelity))
        self.fixed_fidelity = fixed_fidelity
        self.fidelity_update = self.FidelityStrategy[fidelity_option]
        self.scale = scale  # scale for calculate iterations: iterations = self.fidelity * self.scale

    def get_optimum(self):
        return array([zeros(self.dimension)]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        self.eval_count += 1
        pass

    """Fidelity Func"""
    def fidelity_fixed(self, gen, n_gen):
        r"""f(x) = n

        :param gen: Current generation of optimizer
        :param n_gen: Total generations of optimizer
        :return: fidelity = self.fixed_fidelity  \in [1, self.max_fidelity]
        """
        self.fidelity = self.fixed_fidelity

    def fidelity_relu(self, gen, n_gen):
        r"""Linear Unit: f(x) = x, x \in [0, 1]

        :param gen: Current generation of optimizer
        :param n_gen: Total generations of optimizer
        :return: fidelity = (gen / n_gen) * self.max_fidelity
        """
        x = gen / n_gen
        self.fidelity = ceil(x * self.max_fidelity)

    def fidelity_sigmoid(self, gen, n_gen):
        r"""Logistic Func: f(x) = sigmoid(x) = 1 / (1 + e^(-x))

        :param gen: Current generation of optimizer
        :param n_gen: Total generations of optimizer
        :return: fidelity = sigmoid(10 * gen / n_gen - 5) * self.max_fidelity
        """
        x = 10 * gen / n_gen - 5
        self.fidelity = ceil(1 / (1 + exp(-x)) * self.max_fidelity)

    def fidelity_sin(self, gen, n_gen):
        r""" Sin Func: f(x) = sin(x), x \in [0, pi/2]

        :param gen: Current generation of optimizer
        :param n_gen: Total generations of optimizer
        :return: fidelity = sin(gen / n_gen * pi / 2) * self.max_fidelity
        """
        x = gen / n_gen * pi / 2
        self.fidelity = ceil(sin(x) * self.max_fidelity)

    def fidelity_power(self, gen, n_gen):
        r""" Power Func: f(x) = x^n, x \in [0, 1], e.g. n=2

        :param gen: Current generation of optimizer
        :param n_gen: Total generations of optimizer
        :return: fidelity = power(gen / n_gen, 2) * self.max_fidelity
        """
        x = gen / n_gen
        self.fidelity = ceil(power(x, 2) * self.max_fidelity)
