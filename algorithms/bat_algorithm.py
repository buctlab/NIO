from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, asarray, apply_along_axis, where, zeros, exp, append
import logging

logging.basicConfig()
logger = logging.getLogger('BA')
logger.setLevel('INFO')


class BatAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.population \in [15, 50], n = 20 for low dim or n = 40 for high dim by default

        # self.A = kwargs.pop('A', 0.5)  # loudness \in [1, 2]
        # self.r = kwargs.pop('r', 0.5)  # pulse rate \in [0, 1]
        self.alpha = kwargs.pop('alpha', 0.9)  # \in [0, 1]
        self.gama = kwargs.pop('gama', 0.9)  # \in [0, inf]
        self.f_min = kwargs.pop('f_min', 0.0)  # minimum frequency = 0
        self.f_max = kwargs.pop('f_max', 2.0)  # maximum frequency \in [ ,100]

    def adjust_frequency(self):
        # beta = self.Rand.uniform(0, 1, [self.population, self.dim])
        # frequency = self.f_min + (self.f_max - self.f_min) * beta
        # return frequency
        return self.Rand.uniform(self.f_min, self.f_max, [self.population, self.dim])

    def update_velocity(self, velocity, bat_pos, best_pos, frequency):
        return velocity + (bat_pos - best_pos) * frequency

    def generate_position(self, bat_pos, velocity):
        return bat_pos + velocity

    def random_walk(self, bat_pos, best_pos, rate, A_avg):
        if self.Rand.rand() > rate:
            # Matlab implement
            # The factor 0.001 limits the step sizes of random walks
            # return best_pos + 0.001 * self.Rand.normal(0, 1, self.dim)
            epsilon = self.Rand.uniform(-1, 1, self.dim)
            return best_pos + epsilon * A_avg
        else:
            return bat_pos

    def increase_pulse_rate(self, rate):
        r"""iter: 1->max, 1-exp(-self.gama*self.iter): 0->1, rate: 0->self.r

        :return: Pulse rate at current iter
        """
        return rate * (1 - exp(-self.gama * self.iter))

    def reduce_loudness(self, A):
        return self.alpha * A

    def update_solution(self, pos, fit, pos_new, fit_new, rate, A):
        iu = where(fit_new < fit)
        for i in iu[0]:
            if self.Rand.rand() < A[i]:
                pos[i], fit[i] = pos_new[i], fit_new[i]
                rate[i] = self.increase_pulse_rate(self.r[i])
                A[i] = self.reduce_loudness(A[i])
        return pos, fit, rate, A

    def run(self):
        bat_location = self.initial_position()
        bat_fitness = apply_along_axis(self.cost_function, 1, bat_location)

        best_index = argmin(bat_fitness)
        best_sol, best_val = bat_location[best_index], bat_fitness[best_index]

        velocity = zeros([self.population, self.dim])

        # rate = full(self.population, self.r)  # self.r: r_i^0 = 0 or \in [0, 1]
        # A = full(self.population, self.A)

        rate = self.Rand.uniform(0, 1, self.population)
        self.r = rate.copy()
        A = self.Rand.uniform(1, 2, self.population)  # A_i^0 \in [1, 2]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = bat_location
            self.iter_solution.loc[self.iter] = append(best_sol, best_val)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[self.iter].to_dict()))

            frequency = self.adjust_frequency()
            velocity = self.update_velocity(velocity, bat_location, best_sol, frequency)
            bat_loc_new = self.generate_position(bat_location, velocity)
            # Generate a new solution by flying randomly
            A_avg = A.mean()
            bat_loc_new = asarray([self.random_walk(bat_loc_new[i], best_sol, rate[i], A_avg) for i in range(self.population)])
            bat_loc_new = apply_along_axis(self.boundary_handle, 1, bat_loc_new)
            bat_fit_new = apply_along_axis(self.cost_function, 1, bat_loc_new)
            bat_location, bat_fitness, rate, A = self.update_solution(bat_location, bat_fitness, bat_loc_new, bat_fit_new, rate, A)

            ibest = argmin(bat_fitness)
            if bat_fitness[ibest] < best_val:
                best_sol, best_val = bat_location[ibest], bat_fitness[ibest]

        self.best_solution.iloc[:] = append(best_sol, best_val)
        return best_sol, best_val


if __name__ == '__main__':
    ba = BatAlgorithm(func=Ackley(), population=30, iterations=200, stopping_eval=2999,debug=True)
    best_sol, best_val = ba.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(ba.eval_count)

