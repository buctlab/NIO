from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, asarray, apply_along_axis, where, zeros, exp, append
from scipy.stats import levy
import logging

logging.basicConfig()
logger = logging.getLogger('FPA')
logger.setLevel('INFO')


class FlowerPollinationAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.P = kwargs.pop('P', 0.8)

    def get_best(self, population):
        min_fitness = self.cost_function(population[0])
        flag = 0
        for i in range(len(population)):
            if self.cost_function(population[i]) < min_fitness:
                min_fitness = self.cost_function(population[i])
                flag = i
        best = population[flag]
        return best

    def global_pollination(self, popu, best_value):
        pop = []
        step = []
        for i in range(self.func.dimension):
            step.append(levy.pdf(1.5))
        for i in range(self.func.dimension):
            temp = popu[i] + step[i] * (best_value[i] - popu[i])
            pop.append(temp)
        return pop

    def local_pollination(self, popu, population):
        pop = []
        alpha = self.Rand.uniform(0, 1)
        index_one = self.Rand.randint(0, self.population - 1)
        index_two = self.Rand.randint(0, self.population - 1)
        while index_one == index_two:
            index_one = self.Rand.randint(0, self.population - 1)
            index_two = self.Rand.randint(0, self.population - 1)
        for i in range(self.func.dimension):
            temp = popu[i] + alpha * (population[index_two][i] - population[index_one][i])
            pop.append(temp)
        return pop

    def run(self):
        flower = self.initial_position()
        best_value = self.get_best(flower)
        best_ind = self.cost_function(best_value)

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = flower
            self.iter_solution.loc[self.iter] = append(best_value, best_ind)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            temp = []
            for i in range(self.func.dimension):
                temp.append(best_value[i])
            temp.append(best_ind)
            for i in range(self.population):
                if self.Rand.rand() < self.P:
                    temp = self.global_pollination(flower[i], best_value)
                else:
                    temp = self.local_pollination(flower[i], flower)
                for j in range(self.func.dimension):
                    if temp[j] < self.lower[j]:
                        temp[j] = self.lower[j]
                    if temp[j] > self.upper[j]:
                        temp[j] = self.upper[j]
                if self.cost_function(temp) < self.cost_function(flower[i]):
                    flower[i] = temp
            best_value = self.get_best(flower)
            best_ind = self.cost_function(best_value)
        self.best_solution.iloc[:] = append(best_value, best_ind)
        return asarray(best_value), best_ind

# TODO: use original


if __name__ == '__main__':
    fpa = FlowerPollinationAlgorithm(func=Ackley(), iterations=200, debug=True)
    best_sol, best_val = fpa.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(fpa.eval_count)
