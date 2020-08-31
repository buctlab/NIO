import logging

from numpy import apply_along_axis, argsort, concatenate, append, argmin

from algorithms.algorithm import Algorithm, Ackley

logging.basicConfig()
logger = logging.getLogger('BWOA')
logger.setLevel('INFO')


class BlackWidowOptimizationAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.procreate_rate = 0.6
        self.cannibalism_rate = 0.44
        self.mutation_rate = 0.4

    def get_female_and_children(self, parent):
        parent = parent.reshape(2, self.dim)

        parent_fit = apply_along_axis(self.cost_function, 1, parent)
        sorted_parent_index = argsort(parent_fit)[0]
        female = parent[sorted_parent_index]
        alpha = self.Rand.rand(int(self.dim / 2), 1)
        ch1 = alpha * parent[0] + (1 - alpha) * parent[1]
        ch2 = alpha * parent[1] + (1 - alpha) * parent[0]

        all_pop_per_generation = concatenate((ch1, ch2, female.reshape(1, self.dim)), 0)
        nc = int(self.cannibalism_rate * all_pop_per_generation.shape[0])
        all_pop_per_generation_fit = apply_along_axis(self.cost_function, 1, all_pop_per_generation)
        index_asc = argsort(-all_pop_per_generation_fit)[nc:]
        return all_pop_per_generation[index_asc]

    def procreating_and_cannibalism(self, pop1):
        parent_index = apply_along_axis(self.replace_same_element, 1,
                                        self.Rand.randint(0, len(pop1), size=(len(pop1), 2)), len(pop1))

        parent = pop1[parent_index]
        parent = parent.reshape([parent.shape[0], parent.shape[1] * parent.shape[2]])
        pop2 = apply_along_axis(self.get_female_and_children, 1, parent)
        return pop2.reshape(pop2.shape[0] * pop2.shape[1], self.dim)

    def get_reproduction_pop1(self, black_widow_pos, black_widow_fit):
        # number of reproduction
        nr = int(self.procreate_rate * len(black_widow_pos))
        index_asc = argsort(black_widow_fit)[:nr]
        return black_widow_pos[index_asc], black_widow_fit[index_asc]

    def get_mutation_number(self, black_widow_pos):
        return int(self.mutation_rate * len(black_widow_pos))

    def swap_in_solution(self, mutate_and_index):
        pop = mutate_and_index[:self.dim]
        pop[int(mutate_and_index[-2])], pop[int(mutate_and_index[-1])] = pop[int(mutate_and_index[-1])], pop[
            int(mutate_and_index[-2])]
        return pop

    def replace_same_element(self, mutate_index, len_array):
        while mutate_index[0] == mutate_index[1]:
            mutate_index[1] = self.Rand.randint(0, len_array)
        return mutate_index

    def mutation(self, nm, pop1):
        mutation_pop_index = self.Rand.choice(len(pop1), nm, replace=False)
        mutate = pop1[mutation_pop_index]
        mutate_index = apply_along_axis(self.replace_same_element, 1,
                                        self.Rand.randint(0, self.dim, size=(len(mutate), 2)), self.dim)
        mutate_and_index = concatenate((mutate, mutate_index), axis=1)
        pop3 = apply_along_axis(self.swap_in_solution, 1, mutate_and_index)
        return pop3

    def get_population_per_generation(self, pop2, pop3):
        pop = concatenate((pop2, pop3), axis=0)
        pop_fit = apply_along_axis(self.cost_function, 1, pop)
        index_asc = argsort(pop_fit)[:self.population]
        return pop[index_asc], pop_fit[index_asc]

    def run(self):
        black_widow_pos = self.initial_position()
        black_widow_fit = apply_along_axis(self.cost_function, 1, black_widow_pos)
        best_index = argmin(black_widow_fit)
        best_widow = black_widow_pos[best_index]
        best_fit = black_widow_fit[best_index]
        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = black_widow_pos
            black_widow_fit = apply_along_axis(self.cost_function, 1, black_widow_pos)

            # pop1: Select the best nr solutions in pop
            pop1, pop1_fit = self.get_reproduction_pop1(black_widow_pos, black_widow_fit)

            self.iter_solution.loc[self.iter] = append(pop1[0], pop1_fit[0])
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))
            # pop2: Procreating and cannibalism
            pop2 = self.procreating_and_cannibalism(pop1)
            # nm: number of mutation children
            nm = self.get_mutation_number(black_widow_pos)
            # pop3: Mutation
            pop3 = self.mutation(nm, pop1)
            black_widow_pos, black_widow_fit = self.get_population_per_generation(pop2, pop3)
            # break

        self.best_solution.iloc[:] = append(black_widow_pos[0], black_widow_fit[0])
        return black_widow_pos[0], black_widow_fit[0]


if __name__ == '__main__':
    bwoa = BlackWidowOptimizationAlgorithm(func=Ackley(), iterations=1, debug=True)
    bwoa.run()
