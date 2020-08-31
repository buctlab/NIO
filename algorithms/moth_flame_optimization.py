from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, zeros, full, inf, apply_along_axis, where, round, concatenate, fabs, exp, cos, pi, argsort, append, argmin
import logging

logging.basicConfig()
logger = logging.getLogger('MFO')
logger.setLevel('INFO')


class MothFlameOptimization(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_flame_no(self, iter):
        r"""Update flames number.

        :param iter: Current iteration times
        :return: Number of flames
        """
        return int(round(self.population - iter * ((self.population - 1) / self.iterations)))

    def generate_a(self, iter):
        r"""Generate a to calculate t

        :param iter: Current iteration times
        :return: a linearly decreases from -1 to -2
        """
        return -1 + iter * ((-1) / self.iterations)

    def update_flame(self, pre_moth_pos, pre_moth_fit, flame_pos, flame_fit):
        # Sort the previous moths and flames
        pos = concatenate((pre_moth_pos, flame_pos), axis=0)
        fit = concatenate((pre_moth_fit, flame_fit), axis=0)
        indexes = argsort(fit)[:self.population]
        return pos[indexes], fit[indexes]

    def update_moth(self, i, flame_no, a, moth_pos, flame_pos):
        D = fabs(flame_pos[i] - moth_pos[i])  # distance to flame
        b = 1  # a constant for defining the shape of the logarithmic spiral
        t = (a - 1) * self.Rand.uniform(0, 1, self.dim) + 1  # a random number in [-1, 1]

        if i <= flame_no:
            return D * exp(b * t) * cos(2 * pi * t) + flame_pos[i]
        else:
            return D * exp(b * t) * cos(2 * pi * t) + flame_pos[flame_no-1]

    def run(self):
        moth_pos = self.initial_position()
        moth_fit = apply_along_axis(self.cost_function, 1, moth_pos)

        # Sort the first population of moths
        idx = argsort(moth_fit)
        flame_pos, flame_fit = moth_pos[idx], moth_fit[idx]

        best_flame_pos, best_flame_fit = flame_pos[0], flame_fit[0]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = moth_pos
            self.iter_solution.loc[self.iter] = append(best_flame_pos, best_flame_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            flame_no = self.update_flame_no(self.iter)

            previous_moth_pos, previous_moth_fit = moth_pos.copy(), moth_fit.copy()

            a = self.generate_a(self.iter)
            moth_pos = asarray([self.update_moth(i, flame_no, a, moth_pos, flame_pos) for i in range(self.population)])

            moth_pos = apply_along_axis(self.boundary_handle, 1, moth_pos)
            moth_fit = apply_along_axis(self.cost_function, 1, moth_pos)

            flame_pos, flame_fit = self.update_flame(previous_moth_pos, previous_moth_fit, flame_pos, flame_fit)
            best_flame_pos, best_flame_fit = flame_pos[0], flame_fit[0]

        self.best_solution.iloc[:] = append(best_flame_pos, best_flame_fit)
        return best_flame_pos, best_flame_fit


if __name__ == '__main__':
    mfo = MothFlameOptimization(func=Ackley(dimension=50),  iterations=5000, debug=False)
    best_sol, best_val = mfo.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    # print(mfo.eval_count)
