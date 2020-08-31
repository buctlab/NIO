from algorithms.algorithm import Algorithm
from numpy import argmin, apply_along_axis, arange, append, argsort, concatenate, where, setdiff1d

import logging

logging.basicConfig()
logger = logging.getLogger('SFO')
logger.setLevel('INFO')


class SailfishOptimizer(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.A = kwargs.pop('a', 4)
        self.e = kwargs.pop('e', 0.001)
        self.pp = kwargs.pop('pp', 0.02)  # The ratio of sailfish to sardines

        #  self.population: the amount of sailfish
        self.s = int(self.population / self.pp)  # the amount of sardine

    def init_population(self, population):
        return self.Rand.uniform(self.func.lower, self.func.upper, [population, self.func.dimension])

    def update_position_sf(self, position, elite_sf, injured_s, best):
        pd = 1 - self.population / (self.s + self.population)
        # r = self.Rand.rand(self.population, self.func.dimension)
        r = self.Rand.rand(1)
        # na: lambda
        na = 2 * r * pd - pd

        r = self.Rand.rand(self.population, self.func.dimension)
        new_position = elite_sf - na * (r * (elite_sf + injured_s) / 2 - position)
        new_position[best] = position[best]  # elite_sf doesn't update
        return new_position

    def update_position_s(self, position, elite_sf):
        ap = self.A * (1 - 2 * self.iter * self.e)

        if ap < 0.5:
            a = int(ap * self.s)
            b = int(ap * self.dim)

            # 𝛼 sardines with 𝛽 variables of sardine will be updated
            self.Rand.shuffle(position)
            variable = arange(self.dim)
            self.Rand.shuffle(variable)
            variable = variable[:b]

            for j in variable:
                r = self.Rand.rand(1)
                position[:a, j] = r * (elite_sf[j] - position[:a, j] + ap)

        else:
            # update all
            r = self.Rand.rand(len(position), self.func.dimension)
            position = r * (elite_sf - position + ap)

        return position

    def better_solution(self, sailfish_fit, sardine_fit, sailfish_pos, sardine_pos):
        new_sailfish_pos = []
        new_sailfish_fit = []

        con_fit = concatenate((sailfish_fit, sardine_fit), 0)
        con_pos = concatenate((sailfish_pos, sardine_pos), 0)

        sort_fit_index = argsort(con_fit)[0:self.population]

        # sailfish更新
        sailfish_fit = con_fit[sort_fit_index]
        sailfish_pos = con_pos[sort_fit_index]

        # sardine更新
        sort_fit_index = sort_fit_index[where(sort_fit_index > self.population)]-self.population
        new_sardine_index = arange(self.s)
        new_sardine_index = setdiff1d(new_sardine_index,sort_fit_index)
        sardine_fit = sardine_fit[new_sardine_index]
        sardine_pos = sardine_pos[new_sardine_index]
        self.s = sardine_pos.shape[0]
        return sailfish_pos, sailfish_fit, sardine_pos, sardine_fit

    def run(self):
        sailfish_pos = self.initial_position()
        sardine_pos = self.init_population(self.s)

        sailfish_fit = apply_along_axis(self.cost_function, 1, sailfish_pos)
        sardine_fit = apply_along_axis(self.cost_function, 1, sardine_pos)

        best_sf = argmin(sailfish_fit)
        best_s = argmin(sardine_fit)

        elite_sf = sailfish_pos[best_sf]
        injured_s = sardine_pos[best_s]

        while not self.stopping_criteria(self.iter):  # end condition
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = sailfish_pos
            self.iter_solution.loc[self.iter] = append(elite_sf, sailfish_fit[best_sf])
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            sailfish_pos = self.update_position_sf(sailfish_pos, elite_sf, injured_s, best_sf)
            sardine_pos = self.update_position_s(sardine_pos, elite_sf)

            sailfish_pos = apply_along_axis(self.boundary_handle, 1, sailfish_pos)
            sardine_pos = apply_along_axis(self.boundary_handle, 1, sardine_pos)

            sardine_fit = apply_along_axis(self.cost_function, 1, sardine_pos)
            sailfish_fit = apply_along_axis(self.cost_function, 1, sailfish_pos)

            sailfish_pos, sailfish_fit, sardine_pos, sardine_fit = self.better_solution(sailfish_fit, sardine_fit,
                                                                                        sailfish_pos, sardine_pos)

            if self.s > 0:
                best_sf = argmin(sailfish_fit)
                best_s = argmin(sardine_fit)
                elite_sf = sailfish_pos[best_sf]
                injured_s = sardine_pos[best_s]
        self.best_solution.iloc[:] = append(elite_sf, sailfish_fit[best_sf])
        return elite_sf, sailfish_fit[best_sf]


if __name__ == '__main__':

    from benchmarks import *

    SFO = SailfishOptimizer(iterations=200, population=30, func=Ackley(), debug=True)
    print(SFO.run())
    # logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    #
    # from visualizer import PlotConvergence
    # import pandas as pd
    # import os
    #
    # convergence_data = pd.DataFrame(columns=['SailfishOptimizer'], index=list(range(1, SFO.iterations + 1)))
    # convergence_data['SailfishOptimizer'] = SFO.iter_solution['Fitness']
    # convergence_fig_path = r"output/PlotConvergence/Ackley_SFO.png"
    # (path, filename) = os.path.split(convergence_fig_path)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # convergence = PlotConvergence(data=convergence_data, benchmark=Michalewicz(dimension=5), path=convergence_fig_path,
    #                               show=True)
    # convergence.plot_convergence()
