from algorithms.algorithm import Algorithm, Ackley
from numpy import apply_along_axis, argmin, argmax, sum, full, inf, asarray, mean, where, sqrt, fabs, zeros, append
from scipy.spatial.distance import euclidean as ed
import logging

logging.basicConfig()
logger = logging.getLogger('KH')
logger.setLevel('INFO')


class KrillHerdBase(Algorithm):

    def __init__(self, **kwargs):
        # self.population = kwargs.setdefault('population', 25)  # 25 in paper
        super().__init__(**kwargs)

        self.N_max = kwargs.pop('N_max', 0.01)  # the maximum induced speed
        self.V_f = kwargs.pop('V_f', 0.02)  # the foraging speed
        self.D_max = kwargs.pop('D_max', 0.002)  # [0.002, 0.010] the maximum diffusion speed

        self.C_t = kwargs.pop('C_t', 0.93)  # constant $\in [0, 2]$ (default value is 0.5 in the article)
        self.w_n = kwargs.pop('W_n', 0.42)  # [0, 1] the inertia weights of the motion induced
        self.w_f = kwargs.pop('W_f', 0.38)  # [0, 1] the inertia weights of the foraging motion

        # self.d_s = kwargs.pop('d_s', 2.63)  # maximum euclidean distance for neighbors
        self.nn = kwargs.pop('nn', 5)  # maximum neighbors for neighbors effect

        self._Cr = kwargs.pop('_Cr', 0.2)  # Crossover rate
        self._Mu = kwargs.pop('_Mu', 0.05)  # Mutation rate
        self.epsilon = kwargs.pop('epsilon', 1e-31)  # Small numbers for division

        self.bRange = fabs(self.upper - self.lower)

    def init_weights(self):
        r"""the inertia weights (w_n, w_f) generate by user-defined.

        :return: w_n or w_f \in [0, 1]
        """
        return full(self.dim, self.w_n), full(self.dim, self.w_f)

    def sens_range(self, i, kh_pos):
        r"""Get the sensing distance for ith krill individual.

        :param i: Index of ith krill individual
        :param kh_pos: The position of krill herd
        :return: The sensing distance for ith krill individual
        """
        return sum([ed(kh_pos[i], kh_pos[j]) for j in range(self.population)]) / (self.nn * self.population)

    def get_neighbors(self, i, ids, kh_pos):
        r"""Get neighbors of ith krill individual.

        :param i: Index of ith krill individual
        :param ids: Sensing distance of ith krill individual
        :param kh_pos: The position of krill herd
        :return: {list} - Neighbors of ith krill individual.
        """
        neighbors = list()
        n = 0
        for j in range(self.population):
            if n < self.nn and j != i and ed(kh_pos[i], kh_pos[j]) < ids:
                n += 1
                neighbors.append(j)
        return neighbors

    def boundary_handle(self, x):
        r"""Repair solution and put the solution in the bounds of problem.

        :param x: Solution to repair
        :return: Repaired solution within the search space
        """
        ir = where(x > self.upper)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        ir = where(x < self.lower)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        return x

    def get_food_position(self, kh_pos, kh_fit):
        r"""Get position of the virtual center of food and its fitness.
        The virtual center of food concentration is estimated according to the fitness distribution of the krill individuals, which is inspired from “center of mass”.

        :param kh_pos: The position of krill herd
        :param kh_fit: The fitness of krill herd
        :return: Position and fitness of the virtual center
        """
        food_pos = self.boundary_handle(sum(kh_pos / (kh_fit + self.epsilon).reshape(self.population, 1), axis=0) / sum(1 / (kh_fit + self.epsilon)))
        # food_pos = self.repair(asarray([sum(kh_pos[:, i] / (kh_fit + self.epsilon)) for i in range(self.dim)]) / sum(1 / (kh_fit + self.epsilon)))
        food_fit = self.cost_function(food_pos)
        return food_pos, food_fit

    def fun_x(self, x, y):
        return ((y - x) + self.epsilon) / (ed(y, x) + self.epsilon)

    def fun_k(self, x, y, b, w):
        return ((x - y) + self.epsilon) / ((w - b) + self.epsilon)

    def elitist_select(self, kh_pos, kh_fit, kh_elitist_pos, kh_elitist_fit):
        r"""Update the best previously visited position and previous fitness of each krill individual.

        :param kh_pos: Current position of krill herd
        :param kh_fit: Current fitness of krill herd
        :param kh_elitist_pos: Current best previously visited position of each krill individual
        :param kh_elitist_fit: Current best fitness of each krill individual
        :return: New best previously visited position and previous fitness of each krill individual
        """
        ipb = where(kh_elitist_fit >= kh_fit)
        kh_elitist_pos[ipb], kh_elitist_fit[ipb] = kh_pos[ipb], kh_fit[ipb]
        return kh_elitist_pos, kh_elitist_fit

    def induce_neighbors_motion(self, i, n_old, w_n, kh_pos, kh_fit, ikh_best, ikh_worst):
        r"""Movement induced by other krill individuals.

        :param i: Index of ith krill individual
        :param n_old: The last motion induced of ith krill individual
        :param w_n: The inertia weight of the motion induced
        :param kh_pos: The position of krill herd
        :param kh_fit: The fitness of krill herd
        :param ikh_best: Index of krill individual with best fitness
        :param ikh_worst: Index of krill individual with worst fitness
        :return: New motion induced for ith krill individual
        """
        neighbor_index = self.get_neighbors(i, self.sens_range(i, kh_pos), kh_pos)
        neighbor_pos, neighbor_fit, fit_best, fit_worst = kh_pos[neighbor_index], kh_fit[neighbor_index], kh_fit[ikh_best], kh_fit[ikh_worst]
        # alpha_local: the local effect provided by the neighbors
        alpha_l = sum(asarray([self.fun_k(kh_fit[i], j, fit_best, fit_worst) for j in neighbor_fit]).reshape(len(neighbor_fit), 1) * asarray([self.fun_x(kh_pos[i], j) for j in neighbor_pos]), axis=0) if len(neighbor_index) else 0
        # alpha_target: the target direction effect provided by the best krill individual
        alpha_t = 2 * (1 + self.Rand.rand() * self.iter / self.iterations) * self.fun_k(kh_fit[i], kh_fit[ikh_best], kh_fit[ikh_best], kh_fit[ikh_worst]) * self.fun_x(kh_pos[i], kh_pos[ikh_best]) if kh_fit[ikh_best] < kh_fit[i] else 0
        return self.N_max * (alpha_l + alpha_t) + w_n * n_old

    def induce_foraging_motion(self, i, f_old, w_f, food_pos, food_fit, kh_pos, kh_fit, kh_opt_pos, kh_opt_fit, ikh_best, ikh_worst):
        r"""Foraging motion.

        :param i: Index of ith krill individual
        :param f_old: The last foraging motion
        :param w_f: The inertia weight of the foraging motion
        :param food_pos: The position of food
        :param food_fit: The fitness of food
        :param kh_pos: The position of krill herd
        :param kh_fit: The fitness of krill herd
        :param kh_opt_pos: The best position of each krill so far
        :param kh_opt_fit: The best fitness of each krill so far
        :param ikh_best: Index of krill individual with best fitness
        :param ikh_worst: Index of krill individual with worst fitness
        :return: New foraging motion
        """
        # beta_food: the food attractive
        beta_f = 2 * (1 - self.iter / self.iterations) * self.fun_k(kh_fit[i], food_fit, kh_fit[ikh_best], kh_fit[ikh_worst]) * self.fun_x(kh_pos[i], food_pos) if food_fit < kh_fit[i] else 0
        # beta_best: the effect of the best fitness of the ith krill so far
        beta_b = self.fun_k(kh_fit[i], kh_opt_fit[i], kh_fit[ikh_best], kh_fit[ikh_worst]) * self.fun_x(kh_pos[i], kh_opt_pos[i]) if kh_opt_fit[i] < kh_fit[i] else 0
        return self.V_f * (beta_f + beta_b) + w_f * f_old

    def induce_physical_diffusion(self):
        r"""Physical diffusion.

        :return: The physical diffusion of ith krill individual
        """
        return self.D_max * (1 - self.iter / self.iterations) * self.Rand.uniform(-1, 1, self.dim)

    def delta_t(self):
        r"""Get a time interval: a scale factor of the speed vector

        :return: A time interval
        """
        return self.C_t * sum(self.bRange)

    def cr(self, x_fit, y_fit, x_fit_best, x_fit_worst):
        r"""Crossover probability"""
        return self._Cr * self.fun_k(x_fit, y_fit, x_fit_best, x_fit_worst)

    def mu(self, x_fit, y_fit, x_fit_best, x_fit_worst):
        r"""mutation probability"""
        return self._Mu / (self.fun_k(x_fit, y_fit, x_fit_best, x_fit_worst) + self.epsilon)

    def crossover(self, xi, xr, cr):
        return [xr[m] if self.Rand.rand() < cr else xi[m] for m in range(len(xi))]

    def mutation(self, xi, xgbes, mu):
        return [(xgbes[m] + self.Rand.rand()) if self.Rand.rand() < mu else xi[m] for m in range(len(xi))]

    def run(self):
        kh_position = self.initial_position()
        kh_fitness = apply_along_axis(self.cost_function, 1, kh_position)
        kh_elitist_pos, kh_elitist_fit = full([self.population, self.dim], inf), full(self.population, inf)
        ikh_best, ikh_worst = argmin(kh_fitness), argmax(kh_fitness)
        best_pos, best_fit = kh_position[ikh_best], kh_fitness[ikh_best]
        N, F = full(self.population, .0), full(self.population, .0)
        w_n, w_f = self.init_weights()

        self.iter = 0
        while not self.stopping_criteria_eval():
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = kh_position
            self.iter_solution.loc[self.iter] = append(best_pos, best_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            # Get virtual center of food
            food_pos, food_fit = self.get_food_position(kh_position, kh_fitness)
            if food_fit < best_fit:
                best_pos, best_fit = food_pos, food_fit
            # The best previously visited position and its fitness
            kh_elitist_pos, kh_elitist_fit = self.elitist_select(kh_position, kh_fitness, kh_elitist_pos,
                                                                 kh_elitist_fit)

            # Motion Process
            N = asarray(
                [self.induce_neighbors_motion(i, N[i], w_n, kh_position, kh_fitness, ikh_best, ikh_worst) for i in
                 range(self.population)])
            F = asarray(
                [self.induce_foraging_motion(i, F[i], w_f, food_pos, food_fit, kh_position, kh_fitness, kh_elitist_pos,
                                             kh_elitist_fit, ikh_best, ikh_worst) for i in range(self.population)])
            D = asarray([self.induce_physical_diffusion() for i in range(self.population)])
            kh_new_pos = kh_position + self.delta_t() * (N + F + D)

            # Crossover and mutation
            Cr = asarray([self.cr(kh_fitness[i], kh_fitness[ikh_best], kh_fitness[ikh_best], kh_fitness[ikh_worst]) for i in range(self.population)])
            kh_new_pos = asarray([self.crossover(kh_new_pos[i], kh_position[i], Cr[i]) for i in range(self.population)])
            Mu = asarray([self.mu(kh_fitness[i], kh_fitness[ikh_best], kh_fitness[ikh_best], kh_fitness[ikh_worst]) for i in range(self.population)])
            kh_new_pos = asarray([self.mutation(kh_new_pos[i], kh_position[ikh_best], Mu[i]) for i in range(self.population)])

            kh_position = apply_along_axis(self.boundary_handle, 1, kh_new_pos)
            kh_fitness = apply_along_axis(self.cost_function, 1, kh_position)
            ikh_best, ikh_worst = argmin(kh_fitness), argmax(kh_fitness)
            if kh_fitness[ikh_best] < best_fit:
                best_pos, best_fit = kh_position[ikh_best], kh_fitness[ikh_best]

        self.best_solution.iloc[:] = append(best_pos, best_fit)
        return best_pos, best_fit


class KrillHerd(KrillHerdBase):

    def __init__(self, **kwargs):
        KrillHerdBase.__init__(self, **kwargs)

    def get_weights(self):
        r"""the inertia weights (w_n, w_f) are equal to 0.9 at the beginning of the search to emphasize exploration.(used in author's Matlab)

        :return: w_n or w_f \in [0.9->0.1]
        """
        return full(self.dim, 0.1 + 0.8 * (1 - self.iter / self.iterations))

    def get_neighbors(self, i, r):
        r"""Get neighbors of ith krill individual.
        reduce RR cal only once

        :param i: Index of ith krill individual
        :param kh_pos: The position of krill herd
        :return: {list} - Neighbors of ith krill individual.
        """
        neighbors = list()
        n = 0
        ids = mean(r) / self.nn
        for j in range(self.population):
            if r[j] < ids and j != i:
                n += 1
                if n <= self.nn:
                    neighbors.append(j)
        return neighbors

    def induce_neighbors_motion(self, i, n_old, w_n, kh_pos, kh_fit, ikh_best, ikh_worst):
        r"""Movement induced by other krill individuals.

        :param i: Index of ith krill individual
        :param n_old: The last motion induced of ith krill individual
        :param w_n: The inertia weight of the motion induced
        :param kh_pos: The position of krill herd
        :param kh_fit: The fitness of krill herd
        :param ikh_best: Index of krill individual with best fitness
        :param ikh_worst: Index of krill individual with worst fitness
        :return: New motion induced for ith krill individual
        """
        rgb, rr, kw_kgb = kh_pos[ikh_best] - kh_pos[i], kh_pos - kh_pos[i], kh_fit[ikh_worst] - kh_fit[ikh_best]
        r = sqrt(sum(rr * rr, axis=1))
        neighbor_index = self.get_neighbors(i, r)
        # alpha_local: the local effect provided by the neighbors
        alpha_l = 0.0
        for j in neighbor_index:
            alpha_l += (kh_fit[i] - kh_fit[j]) / (kw_kgb + self.epsilon) * (rr[j] / (r[j] + self.epsilon))

        # alpha_target: the target direction effect provided by the best krill individual
        alpha_t = 2 * (1 + self.Rand.rand() * self.iter / self.iterations) * (
                    (kh_fit[i] - kh_fit[ikh_best]) / (kw_kgb + self.epsilon)) * (
                              rgb / (sqrt(sum(rgb * rgb)))) if kh_fit[ikh_best] < kh_fit[i] else 0
        return self.N_max * (alpha_l + alpha_t) + w_n * n_old

    def induce_foraging_motion(self, i, f_old, w_f, food_pos, food_fit, kh_pos, kh_fit, kh_opt_pos, kh_opt_fit, ikh_best, ikh_worst):
        r"""Foraging motion.

        :param i: Index of ith krill individual
        :param f_old: The last foraging motion
        :param w_f: The inertia weight of the foraging motion
        :param food_pos: The position of food
        :param food_fit: The fitness of food
        :param kh_pos: The position of krill herd
        :param kh_fit: The fitness of krill herd
        :param kh_opt_pos: The best position of each krill so far
        :param kh_opt_fit: The best fitness of each krill so far
        :param ikh_best: Index of krill individual with best fitness
        :param ikh_worst: Index of krill individual with worst fitness
        :return: New foraging motion
        """
        rf, kw_kgb = food_pos - kh_pos[i], kh_fit[ikh_worst] - kh_fit[ikh_best]
        # beta_food: the food attractive
        beta_f = 2 * (1 - self.iter / self.iterations) * ((kh_fit[i] - food_fit) / (kw_kgb + self.epsilon)) * (rf / (sqrt(sum(rf * rf)))) if food_fit < kh_fit[i] else 0

        rb = kh_opt_pos[i] - kh_pos[i]
        # beta_best: the effect of the best fitness of the ith krill so far
        beta_b = ((kh_fit[i] - kh_opt_fit[i]) / (kw_kgb + self.epsilon)) * (rb / (sqrt(sum(rb * rb)))) if kh_opt_fit[i] < kh_fit[i] else 0
        return self.V_f * (beta_f + beta_b) + w_f * f_old

    def delta_t(self):
        r"""Get a time interval: (ref author's Matlab code)

        :return: A time interval
        """
        return mean(self.bRange) / 2

    def cr(self, x_fit, x_fit_best, x_fit_worst):
        r"""Crossover probability(used in author's Matlab)"""
        return 0.8 + 0.2 * self.fun_k(x_fit, x_fit_best, x_fit_best, x_fit_worst)

    def run(self):
        kh_position = self.initial_position()
        kh_fitness = apply_along_axis(self.cost_function, 1, kh_position)
        kh_elitist_pos, kh_elitist_fit = full([self.population, self.dim], inf), full(self.population, inf)
        ikh_best, ikh_worst = argmin(kh_fitness), argmax(kh_fitness)
        best_pos, best_fit = kh_position[ikh_best], kh_fitness[ikh_best]
        N, F = full(self.population, .0), full(self.population, .0)

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = kh_position
            self.iter_solution.loc[self.iter] = append(best_pos, best_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            # Get virtual center of food
            food_pos, food_fit = self.get_food_position(kh_position, kh_fitness)
            if food_fit < best_fit:
                best_pos, best_fit = food_pos, food_fit
            # The best previously visited position and its fitness
            kh_elitist_pos, kh_elitist_fit = self.elitist_select(kh_position, kh_fitness, kh_elitist_pos,
                                                                 kh_elitist_fit)

            # Motion Process
            w = self.get_weights()
            N = asarray([self.induce_neighbors_motion(i, N[i], w, kh_position, kh_fitness, ikh_best, ikh_worst) for i in range(self.population)])
            F = asarray([self.induce_foraging_motion(i, F[i], w, food_pos, food_fit, kh_position, kh_fitness, kh_elitist_pos, kh_elitist_fit, ikh_best, ikh_worst) for i in range(self.population)])
            # D = asarray([self.induce_physical_diffusion() for i in range(self.population)])
            D = 0

            # Crossover
            Cr = asarray([self.cr(kh_fitness[i], kh_fitness[ikh_best], kh_fitness[ikh_worst]) for i in range(self.population)])
            kh__new_pos = asarray([self.crossover(kh_position[self.Rand.randint(self.population)], kh_position[i], Cr[i]) for i in range(self.population)])

            kh_new_pos = kh__new_pos + self.delta_t() * (N + F + D)

            kh_position = apply_along_axis(self.boundary_handle, 1, kh_new_pos)
            kh_fitness = apply_along_axis(self.cost_function, 1, kh_position)
            ikh_best, ikh_worst = argmin(kh_fitness), argmax(kh_fitness)
            if kh_fitness[ikh_best] < best_fit:
                best_pos, best_fit = kh_position[ikh_best], kh_fitness[ikh_best]

        self.best_solution.iloc[:] = append(best_pos, best_fit)
        return best_pos, best_fit


if __name__ == '__main__':
    kh = KrillHerd(func=Ackley(), iterations=100,debug=True)
    best_sol, best_val = kh.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(kh.eval_count)
