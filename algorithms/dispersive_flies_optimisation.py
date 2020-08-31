from algorithms.algorithm import Algorithm, Ackley
from numpy import append
import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger('DFO')
logger.setLevel('INFO')


class DispersiveFliesOptimisation(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dt = kwargs.pop('dt', 0.2)

    def initial_position_trans(self):
        position = pd.DataFrame(self.initial_position())
        position['Fitness'] = position.apply(lambda x: self.cost_function(x), axis=1)
        return position

    def best_fly(self, position):
        return position.iloc[position['Fitness'].idxmin(), :].copy(deep=True)

    def update_position(self, position, neighbour_best, swarm_best, fly=0):
        for j in range(0, position.shape[1] - 1):
            # r =
            r = self.Rand.uniform(0, 1)
            position.iloc[fly, j] = neighbour_best[j] + r * (swarm_best[j] - position.iloc[fly, j])
            if position.iloc[fly, j] > self.upper[j]:
                position.iloc[fly, j] = self.upper[j]
            elif position.iloc[fly, j] < self.lower[j]:
                position.iloc[fly, j] = self.lower[j]
        position.iloc[fly, -1] = self.cost_function(position.iloc[fly, 0:position.shape[1] - 1])
        return position

    def run(self):
        population = self.initial_position_trans()

        neighbour_best = self.best_fly(population)
        swarm_best = self.best_fly(population)

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = population.drop(['Fitness'], axis=1).values
            self.iter_solution.loc[self.iter] = swarm_best.values
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            for i in range(0, self.population):
                population = self.update_position(population, neighbour_best, swarm_best, fly=i)
                for j in range(self.func.dimension):
                    r = self.Rand.uniform(0, 1)
                    if r < self.dt:
                        population.iloc[i, j] = self.lower[j] + r * (
                                self.upper[j] - self.lower[j])
                population.iloc[i, -1] = self.cost_function(population.iloc[i, 0:population.shape[1] - 1])

            neighbour_best = self.best_fly(population)
            if swarm_best['Fitness'] > neighbour_best['Fitness']:
                swarm_best = neighbour_best.copy(deep=True)
        self.best_solution.iloc[:] = append(swarm_best.values[0:-1], swarm_best.values[-1])
        return swarm_best.values[0:-1], swarm_best.values[-1]

# TODO: use original


if __name__ == '__main__':
    dfo = DispersiveFliesOptimisation(func=Ackley(), iterations=200, stopping_eval=2971,debug=True)
    best_sol, best_val = dfo.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(dfo.eval_count)
