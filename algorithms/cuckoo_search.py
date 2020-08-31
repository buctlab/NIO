from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, apply_along_axis, empty_like, where, power, sin, fabs, pi, append, zeros
from scipy.special import gamma
import logging

logging.basicConfig()
logger = logging.getLogger('CS')
logger.setLevel('INFO')


class CuckooSearch(Algorithm):
    def __init__(self, **kwargs):
        # self.population = kwargs.setdefault('population', 25)  # [15, 50], 25 in Matlab
        super().__init__(**kwargs)

        self.pa = kwargs.pop('pa', 0.25)  # Discovery rate of alien eggs/solutions [0, 0.5]
        self.alpha = kwargs.pop('alpha', 1)  # The step size. (alpha>0) Paper recommend value: 1; Matlab: 0.01
        self.lamb = kwargs.pop('lamb', 1.5)  # Levy exponent and coefficient. (lambda must be in [0, 2] or sigma_u<0)


    def levy_flight(self, beta):
        r"""Generate step from levy distribution.
        For details, see equation (2.21), Page 16 (chapter 2) of the book X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).

        :param beta: Levy exponent and coefficient
        :return: np.array (ex. [ 1.37861233 -1.49481199  1.38124823])
        """
        sigma_u = power((gamma(1 + beta) * sin(pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2)), 1 / beta)
        sigma_v = 1
        u = self.Rand.normal(0, sigma_u, size=self.dim)
        v = self.Rand.normal(0, sigma_v, size=self.dim)
        step = u / power(fabs(v), 1 / beta)
        return step


    def get_cuckoo(self, pos, best_pos):
        new_pos = empty_like(pos)
        for i in range(self.population):
            step_size = self.alpha * self.levy_flight(self.lamb) * (pos[i] - best_pos)  # Matlab
            # new_pos[i] = pos[i] + step_size * self.Rand.rand(self.dim)  # Matlab
            step_size = self.alpha * self.levy_flight(self.lamb)  # Paper
            new_pos[i] = pos[i] + step_size  # Paper
        return apply_along_axis(self.boundary_handle, 1, new_pos)

    @staticmethod
    def update_nest(pos, fit, new_pos, new_fit):
        replace_index = where(new_fit < fit)
        pos[replace_index], fit[replace_index] = new_pos[replace_index], new_fit[replace_index]
        return pos, fit


    def abandon_solution(self, pos):
        # Discovered index of cuckoo's egg
        abandon_index = where(self.Rand.rand(self.population) < self.pa)
        # New solution by biased/selective random walks
        order1 = self.Rand.permutation(self.population)
        order2 = self.Rand.permutation(self.population)
        step_size = self.Rand.rand() * (pos[order1] - pos[order2])
        new_pos = pos.copy()
        new_pos[abandon_index] = pos[abandon_index] + step_size[abandon_index]
        return apply_along_axis(self.boundary_handle, 1, new_pos)


    def run(self):
        nest_pos = self.initial_position()
        nest_fit = apply_along_axis(self.cost_function, 1, nest_pos)
        best_index = argmin(nest_fit)
        best_pos, best_fit = nest_pos[best_index], nest_fit[best_index]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = nest_pos
            self.iter_solution.loc[self.iter] = append(best_pos, best_fit)
            if self.debug:
                logger.info(
                    "eval_count:{i}/{eval_counts} - {iter_sol}".format(i=self.iter, eval_counts=self.iterations,
                                                                       iter_sol=self.iter_solution.loc[self.iter][
                                                                           'Fitness']))
            # Generate new solutions
            new_nest_pos = self.get_cuckoo(nest_pos, best_pos)
            new_nest_fit = apply_along_axis(self.cost_function, 1, new_nest_pos)
            nest_pos, nest_fit = self.update_nest(nest_pos, nest_fit, new_nest_pos, new_nest_fit)

            # Discovery and randomization
            new_nest_pos = self.abandon_solution(nest_pos)
            new_nest_fit = apply_along_axis(self.cost_function, 1, new_nest_pos)
            nest_pos, nest_fit = self.update_nest(nest_pos, nest_fit, new_nest_pos, new_nest_fit)



            # Find the best objective so far
            best_index = argmin(nest_fit)
            if nest_fit[best_index] < best_fit:
                best_pos, best_fit = nest_pos[best_index], nest_fit[best_index]

        self.best_solution.iloc[:] = append(best_pos, best_fit)
        return best_pos, best_fit


if __name__ == '__main__':
    # import time
    # start=time.time()
    cs = CuckooSearch(func=Ackley(dimension=100), iterations=200, population=30, debug=False)
    print(cs.run())
    # best_sol, best_val = cs.run()
    # end = time.time()
    # print("origin time cost: , eval_count:", (end-start, cs.eval_count))
    # logger.info("best sol:{sol}, best val:{val}".forma,t(sol=best_sol, val=best_val))
    # swarm_pos = cs.iter_swarm_pos
    # from visualizer.animation import PlotAnimatedScatter
    # pas = PlotAnimatedScatter(swarm_pos, Ackley(), None, show=True)
    # # pas.plot()
    # print(cs.iter_swarm_pos)
