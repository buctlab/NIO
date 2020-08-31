from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, asarray, apply_along_axis, where, zeros, exp, newaxis, transpose, concatenate, sqrt, append
import logging

logging.basicConfig()
logger = logging.getLogger('FA')
logger.setLevel('INFO')


class FireflyAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.pop('alpha', 0.5)
        self.beta = kwargs.pop('beta', 1)
        self.absorption = kwargs.pop('absorption', 0.001)

    def distance(self, firefly_a, firefly_b):
        sum = 0
        for i in range(self.func.dimension):
            sum += (firefly_b[i] - firefly_a[i]) ** 2
        return sqrt(sum)

    def move(self, firefly_a, firefly_b):
        for i in range(self.func.dimension):
            firefly_a[i] += self.beta * exp(-self.absorption * ((self.distance(firefly_a, firefly_b)) ** 2)) * \
                            (firefly_b[i] - firefly_a[i]) + self.alpha * (self.Rand.uniform(0, 1) - 0.5)

    def run(self):
        fireflies_position = self.initial_position()
        fireflies_fitness = apply_along_axis(self.cost_function, 1, fireflies_position)

        connection = concatenate((fireflies_fitness[:, newaxis], fireflies_position), axis=1)
        connection = connection[connection[:, 0].argsort()]
        sorted_fireflies_fitness = transpose(connection[:, 0])
        sorted_fireflies = connection[:, 1:self.func.dimension + 1]
        brightest_firefly_position = sorted_fireflies[0, :]
        brightest_firefly_fitness = sorted_fireflies_fitness[0]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = fireflies_position.copy()
            self.iter_solution.loc[self.iter] = append(brightest_firefly_position, brightest_firefly_fitness)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            for i in range(self.population):
                for j in range(i + 1):
                    I = fireflies_fitness[j] * exp(
                        -self.absorption * self.distance(fireflies_position[i, :], fireflies_position[j, :]) ** 2)
                    if fireflies_fitness[i] > I:  # j更亮
                        self.move(fireflies_position[i, :], fireflies_position[j, :])
                    # else:
                    #     for k in range(self.dim):
                    #         fireflies_position[i,k] = random.random()*(self.ub[k]-self.lb[k])+self.lb[k]
                    fireflies_fitness[i] = self.cost_function(fireflies_position[i, :])

            connection = concatenate((fireflies_fitness[:, newaxis], fireflies_position), axis=1)
            connection = connection[connection[:, 0].argsort()]
            fireflies_fitness = transpose(connection[:, 0])
            fireflies_position = connection[:, 1:self.func.dimension + 1]

            if fireflies_fitness[0] < brightest_firefly_fitness:
                brightest_firefly_fitness = fireflies_fitness[0]
                brightest_firefly_position = fireflies_position[0, :]

        self.best_solution.iloc[:] = append(brightest_firefly_position, brightest_firefly_fitness)
        return brightest_firefly_position, brightest_firefly_fitness

# TODO: use original


if __name__ == '__main__':
    fa = FireflyAlgorithm(func=Ackley(), iterations=200, debug=True)
    best_sol, best_val = fa.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(fa.eval_count)
