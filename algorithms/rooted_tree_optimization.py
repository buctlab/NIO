import logging
# import random

from numpy import apply_along_axis, ndarray, lexsort, zeros, append, argmin

from algorithms.algorithm import Algorithm
from benchmarks.rastrigin import Rastrigin

logging.basicConfig()
logger = logging.getLogger('RT')
logger.setLevel('INFO')


class RootedTreeOptimization(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Rr = kwargs.pop("Rr", 0.4)
        self.Rn = kwargs.pop("Rn", 0.3)
        self.Rc = kwargs.pop("Rc", 0.3)

        self.c1 = kwargs.pop("c1", 1)
        self.c2 = kwargs.pop("c2", 1)
        self.c3 = kwargs.pop("c3", 1)
        self.epsilon = 1e-31

    def ratio_of_fitness(self, position, fitness):
        max_fitness = fitness.max()
        position_with_fitness = ndarray((position.shape[0], position.shape[1] + 1))
        position_with_fitness[:, :-1] = position[:, :].copy()
        for i in range(fitness.shape[0]):
            position_with_fitness[i][-1] = 1 - fitness[i] / (max_fitness + self.epsilon)
        return position_with_fitness

    def update_position(self, rooted_tree_position, rooted_tree_position_with_wetness):
        new_rooted_tree_position = zeros(rooted_tree_position.shape)

        # Rr
        for i in range(0, int(self.Rr * self.population)):
            select_individual = self.Rand.randint(0, self.population - 1)
            for dim in range(rooted_tree_position.shape[1]):
                next_individual = rooted_tree_position_with_wetness[select_individual][dim] + self.c1 * \
                                  rooted_tree_position_with_wetness[select_individual][-1] * \
                                  self.Rand.uniform(-1, 1) * abs(self.upper[dim] - self.lower[dim]) / self.iter
                new_rooted_tree_position[i][dim] = next_individual

        # Rn
        for i in range(int(self.Rr * self.population), int((self.Rr + self.Rn) * self.population)):
            for dim in range(rooted_tree_position.shape[1]):
                next_individual = rooted_tree_position_with_wetness[-1][dim] + self.c2 * \
                                  rooted_tree_position_with_wetness[i][-1] * self.Rand.uniform(-1, 1) * abs(
                    self.upper[dim] - self.lower[dim]) / (self.iter * self.population)
                new_rooted_tree_position[i][dim] = next_individual

        # Rc
        for i in range(int((1 - self.Rc) * self.population), self.population):
            for dim in range(rooted_tree_position.shape[1]):
                next_individual = rooted_tree_position_with_wetness[i][dim] + self.c3 * \
                                  rooted_tree_position_with_wetness[i][-1] * self.Rand.uniform(0, 1) * (
                                          rooted_tree_position_with_wetness[-1][dim] -
                                          rooted_tree_position_with_wetness[i][dim])
                new_rooted_tree_position[i][dim] = next_individual
        return new_rooted_tree_position

    def run(self):
        rooted_tree_position = self.initial_position()
        rooted_tree_fitness = apply_along_axis(self.cost_function, 1, rooted_tree_position)
        rooted_tree_position_with_wetness = self.ratio_of_fitness(rooted_tree_position, rooted_tree_fitness)

        best_index = argmin(rooted_tree_fitness)
        best_sol = rooted_tree_position[best_index]
        best_val = rooted_tree_fitness[best_index]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1

            self.iter_swarm_pos.loc[self.iter] = rooted_tree_position.copy()
            self.iter_solution.loc[self.iter] = append(best_sol, best_val)
            # self.best_solution.loc[:] = append(rooted_tree_position_with_wetness[-1, :-1],
            #                                    self.cost_function(rooted_tree_position_with_wetness[-1, :-1]))
            rooted_tree_position_with_wetness = rooted_tree_position_with_wetness[
                lexsort(rooted_tree_position_with_wetness.T[-1, None])]
            if self.debug:
                logging.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                              iter_sol=self.iter_solution.loc[
                                                                                  self.iter].to_dict()))
            rooted_tree_position = self.update_position(rooted_tree_position, rooted_tree_position_with_wetness)
            rooted_tree_position = apply_along_axis(self.boundary_handle, 1, rooted_tree_position)
            rooted_tree_fitness = apply_along_axis(self.cost_function, 1, rooted_tree_position)
            rooted_tree_position_with_wetness = self.ratio_of_fitness(rooted_tree_position, rooted_tree_fitness)

            # update best solution.
            best_i = argmin(rooted_tree_fitness)
            if best_val > rooted_tree_fitness[best_i]:
                # print(rooted_tree_fitness[best_i])
                best_sol, best_val = rooted_tree_position[best_i], rooted_tree_fitness[best_i]

            self.best_solution.iloc[:] = append(best_sol, best_val)
        return best_sol, best_val




if __name__ == '__main__':
    from benchmarks import *
    rt = RootedTreeOptimization(func=Ackley(), iterations=100, debug=True, population=30)
    rt.run()
    print(rt.iter_solution)
