from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, zeros, inf, apply_along_axis, where, fabs, floor, argmin, exp, cos, pi, append
import logging

logging.basicConfig()
logger = logging.getLogger('WOA')
logger.setLevel('INFO')


class WhaleOptimizationAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encircling_prey(self, i, whale_pos, leader_pos, C, A):
        D = fabs(C * leader_pos - whale_pos[i])
        new_whale_pos = leader_pos - A * D
        return new_whale_pos

    def spiral_update_position(self, i, whale_pos, leader_pos, b, l):
        distance = fabs(leader_pos - whale_pos[i])
        new_whale_pos = distance * exp(b * l) * cos(l * 2 * pi) + leader_pos
        return new_whale_pos

    def search_prey(self, i, whale_pos, C, A):
        new_whale_pos = zeros(self.dim)
        for j in range(self.dim):
            rand_leader_index = int(floor(self.population * self.Rand.rand()))
            X_rand = whale_pos[rand_leader_index]
            D = fabs(C * X_rand[j] - whale_pos[i, j])
            new_whale_pos[j] = X_rand[j] - A * D
        return new_whale_pos

    def update_position(self, i, whale_pos, leader_pos, a, a2):
        r1, r2 = self.Rand.rand(), self.Rand.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        b = 1
        l = (a2 - 1) * self.Rand.rand() + 1
        p = self.Rand.rand()
        if p < 0.5:
            if fabs(A) < 1:
                return self.encircling_prey(i, whale_pos, leader_pos, C, A)
            else:
                return self.search_prey(i, whale_pos, C, A)
        else:
            return self.spiral_update_position(i, whale_pos, leader_pos, b, l)

    def run(self):
        whale_pos = self.initial_position()
        whale_fit = apply_along_axis(self.cost_function, 1, whale_pos)

        ibest = argmin(whale_fit)
        leader_pos, leader_fit = whale_pos[ibest], whale_fit[ibest]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = whale_pos
            self.iter_solution.loc[self.iter] = append(leader_pos, leader_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            a = 2 - self.iter * (2 / self.iterations)
            a2 = -1 + self.iter * (-1 / self.iterations)
            whale_pos = asarray([self.update_position(i, whale_pos, leader_pos, a, a2) for i in range(self.population)])

            whale_pos = apply_along_axis(self.boundary_handle, 1, whale_pos)
            whale_fit = apply_along_axis(self.cost_function, 1, whale_pos)

            ibest = argmin(whale_fit)
            if whale_fit[ibest] < leader_fit:
                leader_pos, leader_fit = whale_pos[ibest], whale_fit[ibest]

        self.best_solution.iloc[:] = append(leader_pos, leader_fit)
        return leader_pos, leader_fit


if __name__ == '__main__':
    from benchmarks import *
    woa = WhaleOptimizationAlgorithm(func=Michalewicz(dimension=2), iterations=200, debug=True)
    best_sol, best_val = woa.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
