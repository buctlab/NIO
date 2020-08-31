from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, apply_along_axis, inf, zeros, fabs, argsort, append
import logging

logging.basicConfig()
logger = logging.getLogger('GWO')
logger.setLevel('INFO')


class GreyWolfOptimizer(Algorithm):

    def __init__(self, **kwargs):
        # self.population = kwargs.setdefault('population', 15)  # 5-12 in paper, 30 in Matlab
        super().__init__(**kwargs)

    def update_position(self, wolf_pos, a, alpha_pos, beta_pos, delta_pos):
        r"""Update the Position of search agents including omegas

        :param wolf_pos: Position of a search agent
        :param a: parameter
        :param alpha_pos: Position of the best search agent
        :param beta_pos: Position of the second best search agent
        :param delta_pos: Position of the third best search agent
        :return: New position of current search agent
        """
        r1 = self.Rand.uniform(0, 1, self.dim)
        r2 = self.Rand.uniform(0, 1, self.dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = fabs(C1 * alpha_pos - wolf_pos)
        X1 = alpha_pos - A1 * D_alpha

        r1 = self.Rand.uniform(0, 1, self.dim)
        r2 = self.Rand.uniform(0, 1, self.dim)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = fabs(C2 * beta_pos - wolf_pos)
        X2 = beta_pos - A2 * D_beta

        r1 = self.Rand.uniform(0, 1, self.dim)
        r2 = self.Rand.uniform(0, 1, self.dim)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = fabs(C3 * delta_pos - wolf_pos)
        X3 = alpha_pos - A3 * D_delta

        wolf_pos_new = (X1 + X2 + X3) / 3
        return wolf_pos_new

    def generate_a(self, i):
        r"""Generate new parameter a.

        :param i: current iteration times
        :return: a decreases linearly from 2 to 0
        """
        return 2 - i * (2 / self.iterations)

    @staticmethod
    def update_dominant_wolves(wolf_pos, wolf_fit, alpha_pos, beta_pos, delta_pos, alpha_fit, beta_fit, delta_fit):
        [alpha, beta, delta] = argsort(wolf_fit)[:3]
        for i in [alpha, beta, delta]:
            if wolf_fit[i] < alpha_fit:
                alpha_pos, alpha_fit = wolf_pos[i], wolf_fit[i]
            if alpha_fit < wolf_fit[i] < beta_fit:
                beta_pos, beta_fit = wolf_pos[i], wolf_fit[i]
            if beta_fit < wolf_fit[i] < delta_fit:
                delta_pos, delta_fit = wolf_pos[i], wolf_fit[i]
        return alpha_pos, beta_pos, delta_pos, alpha_fit, beta_fit, delta_fit

    def run(self):
        wolf_pos = self.initial_position()
        wolf_fit = apply_along_axis(self.cost_function, 1, wolf_pos)

        alpha_pos, beta_pos, delta_pos = zeros(self.dim), zeros(self.dim), zeros(self.dim)
        alpha_fit, beta_fit, delta_fit = inf, inf, inf
        alpha_pos, beta_pos, delta_pos, alpha_fit, beta_fit, delta_fit = \
            self.update_dominant_wolves(wolf_pos, wolf_fit, alpha_pos, beta_pos, delta_pos, alpha_fit, beta_fit,
                                        delta_fit)

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = wolf_pos
            self.iter_solution.loc[self.iter] = append(alpha_pos, alpha_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            a = self.generate_a(self.iter)
            wolf_pos = asarray([self.update_position(wolf_pos[i], a, alpha_pos, beta_pos, delta_pos) for i in range(self.population)])
            wolf_pos = apply_along_axis(self.boundary_handle, 1, wolf_pos)
            wolf_fit = apply_along_axis(self.cost_function, 1, wolf_pos)
            alpha_pos, beta_pos, delta_pos, alpha_fit, beta_fit, delta_fit = \
                self.update_dominant_wolves(wolf_pos, wolf_fit, alpha_pos, beta_pos, delta_pos, alpha_fit, beta_fit,
                                            delta_fit)

        self.best_solution.iloc[:] = append(alpha_pos, alpha_fit)
        return alpha_pos, alpha_fit


if __name__ == '__main__':
    from benchmarks import *
    gwo = GreyWolfOptimizer(func=Ackley(dimension=100), iterations=200, debug=True)
    best_sol, best_val = gwo.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    # print(gwo.eval_count)