from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, zeros, inf, apply_along_axis, where, exp, argmin, append
import logging

logging.basicConfig()
logger = logging.getLogger('SSA')
logger.setLevel('INFO')


class SalpSwarmAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_position(self, i, c1, salp_pos, food_pos):
        new_salp_pos = zeros(self.dim)
        if i <= self.population/2:
            c2 = self.Rand.uniform(self.lower, self.upper, self.dim)
            c3 = self.Rand.rand(self.dim)
            index_l, index_g = where(c3 < 0.5), where(c3 >= 0.5)
            new_salp_pos[index_l] = food_pos[index_l] + c1 * c2[index_l]
            new_salp_pos[index_g] = food_pos[index_g] - c1 * c2[index_g]
        elif i < self.population:
            new_salp_pos = (salp_pos[i-1] + salp_pos[i]) / 2
        return new_salp_pos

    def run(self):
        salp_pos = self.initial_position()
        salp_fit = apply_along_axis(self.cost_function, 1, salp_pos)

        ibest = argmin(salp_fit)
        food_pos, food_fit = salp_pos[ibest], salp_fit[ibest]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = salp_pos
            self.iter_solution.loc[self.iter] = append(food_pos, food_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            c1 = 2 * exp(-(4 * self.iter / self.iterations) ** 2)
            salp_pos = asarray([self.update_position(i, c1, salp_pos, food_pos) for i in range(self.population)])
            salp_pos = apply_along_axis(self.boundary_handle, 1, salp_pos)
            salp_fit = apply_along_axis(self.cost_function, 1, salp_pos)

            ibest = argmin(salp_fit)
            if salp_fit[ibest] < food_fit:
                food_pos, food_fit = salp_pos[ibest], salp_fit[ibest]

        self.best_solution.iloc[:] = append(food_pos, food_fit)
        return food_pos, food_fit


if __name__ == '__main__':
    ssa = SalpSwarmAlgorithm(func=Ackley(),  debug=True)
    best_sol, best_val = ssa.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(ssa.eval_count)
