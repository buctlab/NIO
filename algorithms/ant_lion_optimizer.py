from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, zeros, apply_along_axis, cumsum, concatenate, argsort, append
import logging

logging.basicConfig()
logger = logging.getLogger('ALO')
logger.setLevel('INFO')


class AntLionOptimizer(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-31

    def roulette_wheel_selection(self, weights):
        accumulation = cumsum(weights)
        p = self.Rand.rand() * accumulation[-1]
        chosen_index = -1
        for index in range(len(accumulation)):
            if accumulation[index] > p:
                chosen_index = index
                break
        choice = chosen_index
        return choice

    def random_walk_around_antlion(self, antlion, current_iter):
        r"""This function creates random walks.

        :param antlion: Position of ant lions
        :param current_iter: current iteration
        :return: N random walks
        """
        I = 1  # I is the ratio

        if current_iter > self.iterations * 0.95:
            I = 1 + 10 ** 6 * (current_iter / self.iterations)
        elif current_iter > self.iterations * 0.9:
            I = 1 + 10 ** 5 * (current_iter / self.iterations)
        elif current_iter > self.iterations * 0.75:
            I = 1 + 10 ** 4 * (current_iter / self.iterations)
        elif current_iter > self.iterations * 0.5:
            I = 1 + 10 ** 3 * (current_iter / self.iterations)
        elif current_iter > self.iterations * 0.1:
            I = 1 + 10 ** 2 * (current_iter / self.iterations)

        # Decrease boundaries to converge towards antlion
        lb, ub = self.lower / I, self.upper / I
        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
        lb = antlion + lb if self.Rand.rand() < 0.5 else antlion - lb
        ub = antlion + ub if self.Rand.rand() >= 0.5 else antlion - ub

        # This function creates n random walks vectors and normalize according to lb and ub
        RWs = zeros([self.iterations + 1, self.dim])
        for i in range(self.dim):
            X = cumsum(2 * (self.Rand.rand(self.iterations, 1) > 0.5) - 1)
            # [a, b] ---> [c, d]
            a, b = min(X), max(X)
            c, d = lb[i], ub[i]
            X_norm = ((X - a) * (d - c)) / (b - a) + c
            RWs[1:, i] = X_norm
        return RWs

    def update_ant_position(self, iter, sorted_antlion_pos, sorted_antlion_fit, elite_antlion_pos):
        r"""Update position of ant by random walks.

        :param iter: current iteration
        :param sorted_antlion_pos: Position of sorted ant lions
        :param sorted_antlion_fit: Fitness of sorted ant lions
        :param elite_antlion_pos: Position of the best ant lion so far
        :return: A new position of ant
        """
        # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
        roulette_index = self.roulette_wheel_selection(1 / (sorted_antlion_fit + self.epsilon))
        if roulette_index == -1:
            roulette_index = 1

        # RA is the random walk around the selected antlion by roulette wheel
        RA = self.random_walk_around_antlion(sorted_antlion_pos[roulette_index], iter)
        # RE is the random walk around the elite (best antlion so far)
        RE = self.random_walk_around_antlion(elite_antlion_pos, iter)

        ant_pos = (RA[iter] + RE[iter]) / 2
        return ant_pos

    def update_antlion(self, sorted_antlion_pos, sorted_antlion_fit, ant_pos, ant_fit):
        pos = concatenate((sorted_antlion_pos, ant_pos), axis=0)
        fit = concatenate((sorted_antlion_fit, ant_fit), axis=0)
        index_asc = argsort(fit)[:self.population]
        return pos[index_asc], fit[index_asc]

    def run(self):
        antlion_pos = self.initial_position()
        ant_pos = self.initial_position()

        antlion_fit = apply_along_axis(self.cost_function, 1, antlion_pos)
        ant_fit = apply_along_axis(self.cost_function, 1, ant_pos)

        antlion_index_asc = argsort(antlion_fit)
        sorted_antlion_pos, sorted_antlion_fit = antlion_pos[antlion_index_asc], antlion_fit[antlion_index_asc]

        elite_antlion_pos, elite_antlion_fit = sorted_antlion_pos[0], sorted_antlion_fit[0]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = antlion_pos
            self.iter_solution.loc[self.iter] = append(elite_antlion_pos, elite_antlion_fit)
            if self.debug:
                logger.info(
                    "eval_count:{i}/{eval_counts} - {iter_sol}".format(i=self.iter, eval_counts=self.iterations,
                                                                       iter_sol=self.iter_solution.loc[self.iter][
                                                                           'Fitness']))

            # simulate random walks
            ant_pos = asarray(
                [self.update_ant_position(self.iter, sorted_antlion_pos, sorted_antlion_fit, elite_antlion_pos) for i in
                 range(self.population)])
            ant_pos = apply_along_axis(self.boundary_handle, 1, ant_pos)
            ant_fit = apply_along_axis(self.cost_function, 1, ant_pos)
            antlion_pos, antlion_fit = self.update_antlion(sorted_antlion_pos, sorted_antlion_fit, ant_pos, ant_fit)
            # Update the position of elite
            if antlion_fit[0] < elite_antlion_fit:
                elite_antlion_pos, elite_antlion_fit = antlion_pos[0], antlion_fit[0]

            # Keep the elite in the population
            antlion_pos[0], antlion_fit[0] = elite_antlion_pos, elite_antlion_fit
            sorted_antlion_pos, sorted_antlion_fit = antlion_pos, antlion_fit

        self.best_solution.iloc[:] = append(elite_antlion_pos, elite_antlion_fit)
        return elite_antlion_pos, elite_antlion_fit


if __name__ == '__main__':
    from visualizer import PlotSwarmPos2D, PlotConvergence
    import os
    import pandas as pd

    alo = AntLionOptimizer(func=Ackley(), population=30,  debug=True)
    print(alo.run())
    # print(alo.best_solution)
    # print(alo.eval_count)
    # best_sol, best_val = alo.run()
    # logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))

    # swarm_pos_data = pd.DataFrame(alo.iter_swarm_pos)
    # contour_path = r"output/PlotContour/AntLionOptimizer_Ackley.png"
    # (path, filename) = os.path.split(contour_path)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # contour = PlotSwarmPos2D(swarm_pos_data, Ackley(), contour_path, show=True)
    # contour.plot_contour()
    #
    # convergence_data = pd.DataFrame(columns=['AntLionOptimizer'], index=list(range(1, iteration + 1)))
    # convergence_data['AntLionOptimizer'] = alo.iter_solution['Fitness']
    # convergence_fig_path = r"output/PlotConvergence/Ackley.png"
    # (path, filename) = os.path.split(convergence_fig_path)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # convergence = PlotConvergence(data=convergence_data, benchmark=Ackley(), path=convergence_fig_path, show=True)
    # convergence.plot_convergence()
