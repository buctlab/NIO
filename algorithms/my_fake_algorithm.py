'''
ATTENTION: DO NOT USE THIS ALGORITHM!
add a new algorithm: My Fake Algorithm (MFA)

MFA pseudo code:

'''
import numpy as np
from algorithms.algorithm import Algorithm
import logging

# Logging is not forcibly, but it can help you debug.
logging.basicConfig()
logger = logging.getLogger('MFA')
logger.setLevel('INFO')


class MyFakeAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        # call super class get all the necessary parameters.
        super().__init__(**kwargs)

        # switch parameter, range [0, 1]
        # p is a parameter that belongs only to mfa.
        self.p = kwargs.pop("p", 0.5)

    def update_position(self, position):
        new_position = self.Rand.uniform(self.lower, self.upper, [self.population, self.func.dimension])
        return new_position

    def run(self):
        mfa_position = self.initial_position()
        mfa_fitness = np.apply_along_axis(self.cost_function, 1, mfa_position)
        best_individual_index = np.argmin(mfa_fitness)
        best_position = mfa_position[best_individual_index]
        best_fitness = mfa_fitness[best_individual_index]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            # these two parameter must be updated.
            self.iter_swarm_pos.loc[self.iter] = mfa_position
            self.iter_solution.loc[self.iter] = np.append(best_position, best_fitness)

            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            r = self.Rand.uniform()
            if r < self.p:
                mfa_position = self.update_position(mfa_position)
                mfa_fitness = np.apply_along_axis(self.cost_function, 1, mfa_position)
            best_individual_index = np.argmin(mfa_fitness)
            if mfa_fitness[best_individual_index] < best_fitness:
                best_position = mfa_position[best_individual_index]
                best_fitness = mfa_fitness[best_individual_index]

        self.best_solution.iloc[:] = np.append(best_position, best_fitness)
        return best_position, best_fitness


if __name__ == '__main__':
    mfa = MyFakeAlgorithm(iterations=100, debug=True)
    mfa.run()
    print(mfa.best_solution)
