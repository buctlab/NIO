from algorithms.algorithm import Algorithm, Ackley
import pandas as pd
from numpy import argmin, asarray, apply_along_axis, where, full, inf, zeros, clip, fabs, sum, sort, fmin, fmax, empty_like, cumsum, argsort, arange, round, concatenate, append
import logging

logging.basicConfig()
logger = logging.getLogger('GA')
logger.setLevel('INFO')


class GeneticAlgorithm(Algorithm):
    """References: [Book] An Introduction To Genetic Algorithms (  MIT )"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        idx = self.Rand.choice(self.population, self.ts, replace=False)
        best_idx = idx[argmin(fit[idx])]
        return pos[best_idx]

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
        rand = self.Rand.rand(self.dim)
        idx = where(rand < self.cr)
        child1, child2 = parent1.copy(), parent2.copy()
        child1[idx], child2[idx] = parent2[idx], parent1[idx]
        return child1, child2

    def single_point_crossover(self, parent1, parent2):
        n = self.Rand.randint(0, self.dim)
        child1, child2 = parent1.copy(), parent2.copy()
        child1[:n] = parent2[:n]
        child2[n:] = parent1[n:]
        return child1, child2

    def two_point_crossover(self, parent1, parent2):
        n = sort(self.Rand.choice(self.dim, 2))
        child1, child2 = parent1.copy(), parent2.copy()
        child1[n[0]:n[1]] = parent2[n[0]:n[1]]
        child2[n[0]:n[1]] = parent1[n[0]:n[1]]
        return child1, child2

    def arithmetic_crossover(self, parent1, parent2):
        alpha = self.cr + (1 + 2 * self.cr) * self.Rand.rand(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def gaussian_mutation(self, chromosome):
        child = fmin(fmax(self.Rand.normal(chromosome, self.mr * self.bRange), self.lower), self.upper)
        return child

    def uniform_mutation(self, chromosome):
        idx = where(self.Rand.rand(self.dim) < self.mr)
        xm = self.Rand.uniform(self.lower, self.upper)
        child = chromosome.copy()
        child[idx] = xm[idx]
        return child

    def swap_mutation(self, chromosome):
        n = self.Rand.choice(self.dim, 2)
        child = chromosome.copy()
        child[n[0]], child[n[1]] = chromosome[n[1]], chromosome[n[0]]
        return child

    def scramble_mutation(self, chromosome):
        n = sort(self.Rand.choice(self.dim, 2))
        idx = self.Rand.shuffle(arange(n[0], n[1]))
        child = chromosome.copy()
        child[n[0]:n[1]] = chromosome[idx]
        return child

    def inversion_mutation(self, chromosome):
        n = sort(self.Rand.choice(self.dim, 2))
        idx = arange(n[1]-1, n[0]-1)
        child = chromosome.copy()
        child[n[0]:n[1]] = chromosome[idx]
        return child

    def evolve(self, pos, fit):
        n = self.population
        if self.population % 2 == 1:
            n += 1

        children = zeros([self.population, self.dim])
        for i in range(int(n / 2)):
            child1, child2 = self.Crossover(self.Selection(pos, fit), self.Selection(pos, fit))
            child1 = self.Mutation(child1)
            child2 = self.Mutation(child2)
            children[i*2] = child1
            children[i*2+1] = child2
        # logger.info("mutate:{}".format(children))
        children = children[:self.population]
        return children

    def replace_by_sort(self, parent_chr, parent_fit, children_chr, children_fit):
        chromosomes = concatenate((parent_chr, children_chr))
        fitness = concatenate((parent_fit, children_fit))
        index = argsort(fitness)
        offspring_chr = chromosomes[index][:self.population]
        offspring_fit = fitness[index][:self.population]
        return offspring_chr, offspring_fit

    def replace_by_cmp(self, parent_chr, parent_fit, children_chr, children_fit):
        chromosomes, fitness = parent_chr.copy(), parent_fit.copy()
        idx = where(children_fit < parent_fit)
        chromosomes[idx], fitness[idx] = children_chr[idx], children_fit[idx]
        return chromosomes, fitness

    def run(self):
        chromosomes = self.initial_position()
        fitness = apply_along_axis(self.cost_function, 1, chromosomes)
        best_index = argmin(fitness)
        best_sol, best_val = chromosomes[best_index], fitness[best_index]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = chromosomes
            self.iter_solution.loc[self.iter] = append(best_sol, best_val)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))
            # Create offspring
            children_chromosomes = self.evolve(chromosomes, fitness)
            children_fitness = apply_along_axis(self.cost_function, 1, children_chromosomes)
            # Replace the current population with the new population.
            chromosomes, fitness = self.replace_by_cmp(chromosomes, fitness, children_chromosomes, children_fitness)
            # Update global best
            best_index = argmin(fitness)
            if best_val > fitness[best_index]:
                best_sol, best_val = chromosomes[best_index], fitness[best_index]

        self.best_solution.iloc[:] = append(best_sol, best_val)
        return best_sol, best_val


if __name__ == '__main__':
    ga = GeneticAlgorithm(func=Ackley(), population=30, iterations=500, debug=True)
    best_sol, best_val = ga.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(ga.eval_count)
    # from visualizer.animation import PlotSwarmAnimation
    # swarm_pos = ga.iter_swarm_pos
    # pas = PlotSwarmAnimation(swarm_pos, Ackley(), None, show=True)
    # pas.plot()
