import abc
from numpy import asarray, random, where, pi, cos, exp, sqrt, full, zeros, array, arange
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('Algorithm')
logger.setLevel('INFO')


class Algorithm(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        """
        debug:      [True|False]: Show  process of each generation
        func:       [Benchmark]:  Function for calculating
        population: [int]:        Population size
        iterations: [int]:        Count of iteration
        precision:  [float]:      Precision of best_value for stopping criteria
        Rand:                     RandomState with seed=1 by default
        lower & upper:            Boundary of function
        dim:                      Dimension of function
        iter:                     Current Iteration started from 0
        eval_count:               Count of cost function called
        """
        self.debug = kwargs.pop('debug', False)
        # self.evaluations, self.evaluations_num = 0, kwargs.pop('neval', 10000)
        # self.evaluations, self.iterations_num = 0, kwargs.pop('niter', 200)

        self.func = kwargs.pop('func', Ackley())
        self.population = kwargs.pop('population', 30)
        self.iterations = kwargs.pop('iterations', 20)
        self.precision = kwargs.pop('precision', 1e-2)
        self.Rand = random.RandomState(kwargs.pop('seed', 1))  # random generator

        self.lower, self.upper = asarray(self.func.lower), asarray(self.func.upper)
        # self.lower, self.upper = asarray(-100), asarray(100)
        self.dim = self.func.dimension
        self.iter = 0
        self.eval_count = 0
        self.stopping_eval = kwargs.pop('stopping_eval', 200)

        # Use pandas to save csv conveniently
        # self.best_solution: [dim1, dim2, ..., dimN, Fitness]
        self.best_solution = pd.Series(index=arange(1, self.dim + 2))
        self.best_solution.rename(index={self.dim + 1: 'Fitness'}, inplace=True)
        # self.iter_solution: [Iteration1:[dim1, dim2, ..., dimN, Fitness], Iteration2:[...], ..., IterationN:[...]]
        self.iter_solution = pd.DataFrame(index=arange(1, self.iterations + 1), columns=arange(1, self.dim + 2))
        self.iter_solution.rename(columns={self.dim + 1: 'Fitness'}, inplace=True)
        # self.iter_swarm_pos: [Iteration1:[Individual1[dim1, dim2, ..., dimN], ..., IndividualN[...]], ..., IterationN:[Individual1[...], ..., IndividualN[...]]]
        index = pd.MultiIndex.from_product([arange(1, self.iterations + 1), arange(1, self.population + 1)],
                                           names=['Iteration', 'Individual'])
        columns = list(range(self.dim))
        self.iter_swarm_pos = pd.DataFrame(index=index, columns=columns)

    def initial_position(self):
        return self.Rand.uniform(self.lower, self.upper, [self.population, self.func.dimension])

    def boundary_handle(self, x):
        r"""Put the solution in the bounds of problem.

        :param x: Solution to boundary handle
        :return: Bound solution within the search space
        """
        ir = where(x < self.lower)
        x[ir] = self.lower[ir]
        ir = where(x > self.upper)
        x[ir] = self.upper[ir]
        return x

    def cost_function(self, position):
        self.eval_count += 1
        return self.func.eval(position)

    def stopping_criteria(self, i):
        return i >= self.iterations

    def stopping_criteria_precision(self, i, optimum_now):
        if i >= self.stopping_eval:
            return True
        if abs(optimum_now - self.func.get_optimum()[-1]) <= self.precision:
            return True
        return False

    def stopping_criteria_eval(self):
        # return self.eval_count >= self.stopping_eval
        return self.iter >= self.iterations

    @abc.abstractmethod
    def run(self):
        pass

    def run_return_swarm_pos(self):
        self.run()
        return self.iter_swarm_pos

    def run_return_best_val(self):
        self.run()
        return self.best_solution['Fitness']

    def run_return_convergence(self):
        self.run()
        return self.iter_solution['Fitness']

    def run_return_iter_sol(self):
        self.run()
        return self.iter_solution

    def run_return_eval_count(self):
        self.run()
        return self.eval_count


class Ackley:

    def __init__(self, lower=-32.768, upper=32.768, dimension=2):
        self.dimension = dimension
        self.lower = full(self.dimension, lower)
        self.upper = full(self.dimension, upper)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(sol):
        a = 20
        b = 0.2
        c = 2 * pi
        d = len(sol)

        sum1 = 0.0
        sum2 = 0.0
        for i in range(d):
            xi = sol[i]
            sum1 += xi ** 2
            sum2 += cos(c * xi)
        part1 = -a * exp(-b * sqrt(sum1 / d))
        part2 = -exp(sum2 / d)

        return part1 + part2 + a + exp(1)
