from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, asarray, apply_along_axis, where, append
import logging

logging.basicConfig()
logger = logging.getLogger('DE')
logger.setLevel('INFO')


class DifferentialEvolution(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        strategy = {  # evolution strategy
            'CR1': self.cross_rand_1,
            'CB1': self.cross_best_1,
            'CR2': self.cross_rand_2,
            'CB2': self.cross_best_2,
            'CC2R1': self.cross_curr_2_rand_1,
            'CC2B1': self.cross_curr_2_best_1
        }

        self.F = kwargs.pop('F', 2)  # constant factor
        self.CR = kwargs.pop('CR', 0.9)  # crossover rate
        self.CrossMutt = strategy[kwargs.pop('strategy', 'CR1')]

    def cross_rand_1(self, swarm_pos, i, best_pos):
        rand_dim = self.Rand.randint(self.dim)  # the rand_dim dimension cross with mutate individual

        sample = list(range(self.population))
        sample.remove(i)  # r[x] != i
        r = self.Rand.choice(sample, 3, replace=False)

        x = [swarm_pos[r[0]][j] + self.F * (swarm_pos[r[1]][j] - swarm_pos[r[2]][j])
             if self.Rand.rand() < self.CR or j == rand_dim
             else swarm_pos[i][j] for j in range(self.dim)]
        return asarray(x)

    def cross_best_1(self, swarm_pos, i, best_pos):
        rand_dim = self.Rand.randint(self.dim)

        sample = list(range(self.population))
        sample.remove(i)
        r = self.Rand.choice(sample, 2, replace=False)

        x = [best_pos[j] + self.F * (swarm_pos[r[0]][j] - swarm_pos[r[1]][j])
             if self.Rand.rand() < self.CR or j == rand_dim
             else swarm_pos[i][j] for j in range(self.dim)]
        return asarray(x)

    def cross_rand_2(self, swarm_pos, i, best_pos):
        rand_dim = self.Rand.randint(self.dim)

        sample = list(range(self.population))
        sample.remove(i)
        r = self.Rand.choice(sample, 5, replace=False)

        x = [swarm_pos[r[0]][j] + self.F * (swarm_pos[r[1]][j] - swarm_pos[r[2]][j])
             + self.F * (swarm_pos[r[3]][j] - swarm_pos[r[4]][j])
             if self.Rand.rand() < self.CR or j == rand_dim
             else swarm_pos[i][j] for j in range(self.dim)]
        return asarray(x)

    def cross_best_2(self, swarm_pos, i, best_pos):
        rand_dim = self.Rand.randint(self.dim)

        sample = list(range(self.population))
        sample.remove(i)
        r = self.Rand.choice(sample, 4, replace=False)

        x = [best_pos[j] + self.F * (swarm_pos[r[0]][j] - swarm_pos[r[1]][j])
             + self.F * (swarm_pos[r[2]][j] - swarm_pos[r[3]][j])
             if self.Rand.rand() < self.CR or j == rand_dim
             else swarm_pos[i][j] for j in range(self.dim)]
        return asarray(x)

    def cross_curr_2_rand_1(self, swarm_pos, i, best_pos):
        rand_dim = self.Rand.randint(self.dim)

        sample = list(range(self.population))
        sample.remove(i)
        r = self.Rand.choice(sample, 4, replace=False)

        x = [swarm_pos[i][j] + self.F * (swarm_pos[r[0]][j] - swarm_pos[r[1]][j])
             + self.F * (swarm_pos[r[2]][j] - swarm_pos[r[3]][j])
             if self.Rand.rand() < self.CR or j == rand_dim
             else swarm_pos[i][j] for j in range(self.dim)]
        return asarray(x)

    def cross_curr_2_best_1(self, swarm_pos, i, best_pos):
        rand_dim = self.Rand.randint(self.dim)

        sample = list(range(self.population))
        sample.remove(i)
        r = self.Rand.choice(sample, 3, replace=False)

        x = [swarm_pos[i][j] + self.F * (best_pos[j] - swarm_pos[r[0]][j])
             + self.F * (swarm_pos[r[1]][j] - swarm_pos[r[2]][j])
             if self.Rand.rand() < self.CR or j == rand_dim
             else swarm_pos[i][j] for j in range(self.dim)]
        return asarray(x)

    @staticmethod
    def selection(x_pos, x_val, y_pos, y_val):
        return x_pos if x_val < y_val else y_pos

    def boundary_handle(self, x):
        ir = where(x > self.upper)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        ir = where(x < self.lower)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        return x

    def run(self):
        swarm_pos = self.initial_position()
        swarm_val = apply_along_axis(self.cost_function, 1, swarm_pos)

        i_best = argmin(swarm_val)
        best_sol, best_val = swarm_pos[i_best], swarm_val[i_best]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = swarm_pos
            self.iter_solution.loc[self.iter] = append(best_sol, best_val)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            # Mutate / recombine
            new_swarm_pos = asarray([self.CrossMutt(swarm_pos, i, best_sol) for i in range(self.population)])
            new_swarm_pos = apply_along_axis(self.boundary_handle, 1, new_swarm_pos)  # 边界检查
            new_swarm_val = apply_along_axis(self.cost_function, 1, new_swarm_pos)

            # Evaluate / select
            swarm_pos = asarray([self.selection(new_swarm_pos[i], new_swarm_val[i], swarm_pos[i], swarm_val[i]) for i in range(self.population)])
            swarm_val = apply_along_axis(self.cost_function, 1, swarm_pos)

            # Update Global Best
            i_best = argmin(swarm_val)
            if best_val > swarm_val[i_best]:
                best_sol, best_val = swarm_pos[i_best], swarm_val[i_best]

        self.best_solution.iloc[:] = append(best_sol, best_val)
        return best_sol, best_val


if __name__ == '__main__':
    from benchmarks import *
    de = DifferentialEvolution(func=Michalewicz(dimension=10), population=30, iterations=1, debug=True)
    print(de.run())

