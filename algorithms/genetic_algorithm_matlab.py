from algorithms.algorithm import Algorithm, Ackley
import pandas as pd
from numpy import argmin, asarray, apply_along_axis, where, full, inf, zeros, clip, fabs, sum, sort, fmin, fmax, empty_like, cumsum, argsort, arange, round, concatenate, append
import logging

logging.basicConfig()
logger = logging.getLogger('GA')
logger.setLevel('INFO')


class GeneticAlgorithm(Algorithm):
    """References: Goldberg, David E., Genetic Algorithms in Search, Optimization & Machine Learning, Addison-Wesley, 1989."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ec = kwargs.pop("ec", 2)  # Elite count
        self.cf = kwargs.pop("cf", 0.8)  # Crossover fraction

        selection_options = {  # parent_selection options
            'SEL1': self.tournament_selection,
            'SEL2': self.roulette_wheel_selection,
            'SEL3': self.stochastic_uniform_selection,
            'SEL4': self.rank_selection,
            'SEL5': self.random_selection,
        }

        crossover_options = {  # crossover options
            'CR1': self.scattered_crossover,
            'CR2': self.single_point_crossover,
            'CR3': self.two_point_crossover,
            'CR4': self.arithmetic_crossover,
        }

        mutation_options = {  # mutation options
            'MU1': self.gaussian_mutation,
            'MU2': self.uniform_mutation,
            'MU3': self.swap_mutation,
            'MU4': self.scramble_mutation,
            'MU5': self.inversion_mutation,
        }

        self.ts = kwargs.pop('ts', 4)  # tournament size >= 2
        self.mr = kwargs.pop('mr', 0.25)  # mutation rate
        self.cr = kwargs.pop('cr', 0.25)  # crossover rate

        self.Selection = selection_options[kwargs.pop('Selection', 'SEL1')]
        self.Crossover = crossover_options[kwargs.pop('Crossover', 'CR1')]
        self.Mutation = mutation_options[kwargs.pop('Mutation', 'MU1')]

        self.epsilon = 1e-31
        self.bRange = fabs(self.upper - self.lower)

    def roulette_wheel_selection(self, pos, fit):
        """
        First implementations of fitness proportionate selection: Roulette Wheel Selection.
        Note: fitness proportionate selection methods don’t work for cases where the fitness can take a negative value.
        """
        weight = 1 / (fit + self.epsilon)
        accumulation = cumsum(weight)
        p = self.Rand.rand() * accumulation[-1]
        chosen_index = -1
        for index in range(len(accumulation)):
            if accumulation[index] > p:
                chosen_index = index
                break
        return pos[chosen_index]

    def stochastic_uniform_selection(self, pos, fit):
        """Second implementations of fitness proportionate selection: Stochastic Universal Sampling (SUS)"""
        pass

    def tournament_selection(self, pos, fit):
        """
        Tournament Selection can work with negative fitness values.
        """
        idxs = self.Rand.choice(self.population, self.ts, replace=False)
        ibest = idxs[argmin(fit[idxs])]
        return pos[ibest]

    def rank_selection(self, pos, fit):
        order = argsort(-fit)
        ranks = argsort(order) + 1
        accumulation = cumsum(ranks)
        p = self.Rand.rand() * accumulation[-1]
        chosen_index = -1
        for index in range(len(accumulation)):
            if accumulation[index] > p:
                chosen_index = index
                break
        return pos[chosen_index]

    def random_selection(self, pos, fit):
        """this strategy is usually avoided"""
        return pos[self.Rand.randint(0, self.population)]

    def scattered_crossover(self, parent1, parent2):
        """or named uniform_crossover"""
        binary = self.Rand.randint(0, 2, self.dim)  # flip a coin for each chromosome
        idx1, idx2 = where(binary == 1), where(binary == 0)
        # idxs = where(self.Rand.rand(self.dim) < self.cr)  # bias the coin to one parent
        child = empty_like(parent1)
        child[idx1] = parent1[idx1]
        child[idx2] = parent2[idx2]
        return child

    def single_point_crossover(self, parent1, parent2):
        n = self.Rand.randint(0, self.dim)
        child = empty_like(parent1)
        child[:n] = parent1[:n]
        child[n:] = parent2[n:]
        return child

    def two_point_crossover(self, parent1, parent2):
        n = sort(self.Rand.choice(self.dim, 2))
        child = empty_like(parent1)
        child[:n[0]] = parent1[:n[0]]
        child[n[0]:n[1]] = parent2[n[0]:n[1]]
        child[n[1]:] = parent1[n[1]:]
        return child

    def arithmetic_crossover(self, parent1, parent2):
        alpha = self.cr + (1 + 2 * self.cr) * self.Rand.rand(self.dim)
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def gaussian_mutation(self, parent):
        child = fmin(fmax(self.Rand.normal(parent, self.mr * self.bRange), self.lower), self.upper)
        return child

    def uniform_mutation(self, parent):
        idxs = where(self.Rand.rand(self.dim) < self.mr)
        xm = self.Rand.uniform(self.lower, self.upper)
        child = parent.copy()
        child[idxs] = xm[idxs]
        return child

    def swap_mutation(self, parent):
        n = self.Rand.choice(self.dim, 2)
        child = parent.copy()
        child[n[0]], child[n[1]] = parent[n[1]], parent[n[0]]
        return child

    def scramble_mutation(self, parent):
        n = sort(self.Rand.choice(self.dim, 2))
        idxs = self.Rand.shuffle(arange(n[0], n[1]))
        child = parent.copy()
        child[n[0]:n[1]] = parent[idxs]
        return child

    def inversion_mutation(self, parent):
        n = sort(self.Rand.choice(self.dim, 2))
        idxs = arange(n[1]-1, n[0]-1)
        child = parent.copy()
        child[n[0]:n[1]] = parent[idxs]
        return child

    def age_based_kick_selection(self, parents, ages, children):
        idxs = argsort(ages)[-self.ec:][::-1]
        parents[idxs] = children[idxs]
        return parents

    def fitness_based_kick_selection(self, parents, fit, children):
        idxs = argsort(fit)[-self.ec:][::-1]
        parents[idxs] = children[idxs]
        return parents

    def evolve(self, pos, fit, elite_num, cr_num, mu_num):
        elite_children = asarray([self.Selection(pos, fit) for i in range(elite_num)])
        # logger.info("elite:{}".format(elite_children))
        crossover_children = asarray([self.Crossover(self.Selection(pos, fit), self.Selection(pos, fit)) for i in range(cr_num)])
        # logger.info("cross:{}".format(crossover_children))
        mutation_children = asarray([self.Mutation(self.Selection(pos, fit)) for i in range(mu_num)])
        # logger.info("mutate:{}".format(mutation_children))
        children = concatenate((elite_children, crossover_children, mutation_children))
        return children

    def run(self):
        position = self.initial_position()
        fitness = apply_along_axis(self.cost_function, 1, position)
        best_index = argmin(fitness)
        best_sol, best_val = position[best_index], fitness[best_index]
        elite_num = self.ec
        cr_num = int(round(self.population * self.cf))
        mu_num = self.population - self.ec - cr_num

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = position
            self.iter_solution.loc[self.iter] = append(best_sol, best_val)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            position = self.evolve(position, fitness, elite_num, cr_num, mu_num)
            fitness = apply_along_axis(self.cost_function, 1, position)
            best_index = argmin(fitness)
            if best_val > fitness[best_index]:
                best_sol, best_val = position[best_index], fitness[best_index]

        self.best_solution.iloc[:] = append(best_sol, best_val)
        return best_sol, best_val


if __name__ == '__main__':
    ga = GeneticAlgorithm(func=Ackley(), population=30, iterations=200, debug=True)
    best_sol, best_val = ga.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
