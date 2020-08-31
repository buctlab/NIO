from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, argmax, asarray, apply_along_axis, where, fabs, full, append
import logging

logging.basicConfig()
logger = logging.getLogger('WWO')
logger.setLevel('INFO')


class WaterWaveOptimization(Algorithm):

    def __init__(self, **kwargs):
        self.population = kwargs.setdefault('population', 8)  # 5-10
        super().__init__(**kwargs)

        self.hmax = kwargs.pop('hmax', 6)  # the init and reset wave height (Z+) {5, 6} (note: hmax=12 in Matlab)
        self.lamb = kwargs.pop('lamb', 0.5)  # the init wavelength (R+)
        self.alpha = kwargs.pop('alpha', 1.0026)  # the wavelength reduction coefficient [1.001, 1.01]
        # self.beta = kwargs.pop('beta', 0.25)  # the breaking coefficient [0.001, 0.01]

        self.betaMax, self.betaMin = 0.25, 0.001
        self.kMax = min(12, self.dim/2)
        self.bRange = fabs(self.upper - self.lower)
        self.epsilon = 1e-31  # a very small positive number to avoid division-by-zero

    def propagation(self, pos):
        r"""Creates a new wave.

        :param pos: The original position of a wave.
        :return: New position and fitness of a wave
        """
        l = self.bRange
        new_pos = pos + self.Rand.uniform(-1, 1, self.dim) * self.lamb * l
        new_pos = self.boundary_handle(new_pos)
        new_fit = self.cost_function(new_pos)
        return new_pos, new_fit

    def refraction(self, pos, best_pos):
        r"""Perform refraction on waves whose heights decrease to zero.

        :param pos:
        :param best_pos: The best solution found so far
        :return:
        """
        mu = (best_pos + pos) / 2
        sigma = fabs(best_pos - pos) / 2
        new_pos = self.Rand.normal(mu, sigma, self.dim)
        new_pos = self.boundary_handle(new_pos)
        new_fit = self.cost_function(new_pos)
        return new_pos, new_fit

    @staticmethod
    def set_wave_length(wave_length, fit_old, fit):
        r"""Set wavelength after refraction.

        :param wave_length: Wavelength before refraction.
        :param fit_old: Fitness before refraction.
        :param fit: Fitness after refraction.
        :return: Wavelength after refraction.
        """
        return wave_length * fit_old / fit

    def breaking(self, pos, fit, wave_length, beta):
        r"""Perform the breaking operation only on a wave that finds a new best solution

        :param pos: A wave position
        :param fit: A wave fitness
        :param wave_length: A wave wave_length
        :param beta: The breaking coefficient
        :return: The solitary waves are better than global best wave
        """
        k = self.Rand.randint(1, self.kMax+1)
        temp = self.Rand.permutation(self.dim)[:k]
        for i in range(k):
            temp_pos = pos.copy()
            d = temp[i]
            temp_pos[d] = pos[d] + self.Rand.normal(0, 1) * beta * self.bRange[d]
            self.boundary_handle(temp_pos)
            temp_fit = self.cost_function(temp_pos)

            if temp_fit < fit:
                pos[d] = temp_pos[d]
                wave_length = self.set_wave_length(wave_length, fit, temp_fit)
                fit = temp_fit
        return pos, fit, wave_length

    def update_wave_length(self, wave_length, fit, fit_max, fit_min):
        r"""Update the wavelength of each wave.

        :param wave_length: Wavelength respectively to the water wave
        :param fit: Fitness of a water wave
        :param fit_max: The maximum fitness values among the current population
        :param fit_min: The minimum fitness values among the current population
        :return:
        """
        return wave_length * self.alpha ** (-(fit - fit_min + self.epsilon) / (fit_max - fit_min + self.epsilon))

    def update_beta(self):
        r"""Calculate the breaking coefficient."""
        return self.betaMax - (self.betaMax - self.betaMin) * self.iter / self.iterations

    def boundary_handle(self, x):
        r"""If the new position is outside the feasible range, it will be reset to a random position in the range.

        :param x: Position of a water wave
        :return: Boundary inside position of a water wave
        """
        ir = where(x > self.upper)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        ir = where(x < self.lower)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        return x

    def run(self):
        water_wave_pos = self.initial_position()
        water_wave_fit = apply_along_axis(self.cost_function, 1, water_wave_pos)
        wave_length = full(self.population, self.lamb)
        wave_height = full(self.population, self.hmax)
        min_index = argmin(water_wave_fit)
        best_pos, best_fit = water_wave_pos[min_index], water_wave_fit[min_index]
        beta = self.betaMax

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = water_wave_pos
            self.iter_solution.loc[self.iter] = append(best_pos, best_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))
            for i in range(self.population):  # Propagation
                new_pos, new_fit = self.propagation(water_wave_pos[i])
                if new_fit < water_wave_fit[i]:
                    water_wave_pos[i], water_wave_fit[i] = new_pos, new_fit
                    wave_height[i] = self.hmax
                    if new_fit < best_fit and i != min_index:  # Breaking
                        new_pos, new_fit, wave_length[i] = self.breaking(new_pos, new_fit, wave_length[i], beta)
                        best_pos, best_fit = new_pos, new_fit
                else:
                    wave_height[i] -= 1  # decrease wave height
                    if wave_height[i] == 0:  # Refraction
                        fit_old = water_wave_fit[i]
                        water_wave_pos[i], water_wave_fit[i] = self.refraction(water_wave_pos[i], best_pos)
                        wave_height[i] = self.hmax
                        wave_length[i] = self.set_wave_length(wave_length[i], fit_old, water_wave_fit[i])

            min_index, max_index = argmin(water_wave_fit), argmax(water_wave_fit)
            best_pos, best_fit = water_wave_pos[min_index], water_wave_fit[min_index]
            wave_length = asarray([self.update_wave_length(wave_length[i], water_wave_fit[i], water_wave_fit[max_index], water_wave_fit[min_index]) for i in range(self.population)])
            beta = self.update_beta()

        self.best_solution.iloc[:] = append(best_pos, best_fit)
        return best_pos, best_fit


if __name__ == '__main__':
    wwo = WaterWaveOptimization(func=Ackley(), iterations=200, debug=True)
    best_sol, best_val = wwo.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
