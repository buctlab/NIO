from algorithms.algorithm import Algorithm, Ackley
from numpy import argmin, asarray, apply_along_axis, where, full, inf, zeros, append
import logging

logging.basicConfig()
logger = logging.getLogger('PSO')
logger.setLevel('INFO')


class ParticleSwarmOptimization(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.c1 = kwargs.pop('c1', 2.0)  # cognitive component
        self.c2 = kwargs.pop('c2', 2.0)  # social component
        self.w = kwargs.pop('w', 0.7)  # inertia weight
        self.v_max = kwargs.pop('v_max', 4)  # maximal velocity
        self.v_min = kwargs.pop('v_min', -self.v_max)  # minimal velocity

        self.v_min, self.v_max = full(self.dim, self.v_min), full(self.dim, self.v_max)

    def initial_position(self):
        return super(ParticleSwarmOptimization, self).initial_position()

    def update_previous_best(self, particle_pos, particle_fit, pbest_pos, pbest_fit):
        i_pbest = where(pbest_fit > particle_fit)
        pbest_pos[i_pbest], pbest_fit[i_pbest] = particle_pos[i_pbest], particle_fit[i_pbest]
        return pbest_pos, pbest_fit

    def update_velocity(self, velocity, particle_pos, pbest_pos, gbest_pos):
        return self.w * velocity + self.c1 * self.Rand.rand(self.population, self.dim) * (pbest_pos - particle_pos) + \
               self.c2 * self.Rand.rand(self.population, self.dim) * (gbest_pos - particle_pos)

    def update_position(self, particle_pos, velocity):
        return particle_pos + velocity

    def velocity_boundary_handle(self, x, lower, upper):
        ir = where(x < lower)
        x[ir] = lower[ir]
        ir = where(x > upper)
        x[ir] = upper[ir]
        return x

    def run(self):
        particle_pos = self.initial_position()
        particle_fit = apply_along_axis(self.cost_function, 1, particle_pos)
        ibest = argmin(particle_fit)
        pbest_pos, pbest_fit = particle_pos, particle_fit  # previous best
        gbest_pos, gbest_fit = particle_pos[ibest], particle_fit[ibest]  # global best
        velocity = zeros([self.population, self.dim])

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = particle_pos
            self.iter_solution.loc[self.iter] = append(gbest_pos, gbest_fit)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            pbest_pos, pbest_fit = self.update_previous_best(particle_pos, particle_fit, pbest_pos, pbest_fit)

            velocity = self.update_velocity(velocity, particle_pos, pbest_pos, gbest_pos)
            velocity = apply_along_axis(self.velocity_boundary_handle, 1, velocity, self.v_min, self.v_max)

            particle_pos = self.update_position(particle_pos, velocity)

            particle_pos = apply_along_axis(self.boundary_handle, 1, particle_pos)
            particle_fit = apply_along_axis(self.cost_function, 1, particle_pos)
            ibest = argmin(particle_fit)
            if gbest_fit > particle_fit[ibest]:
                gbest_pos, gbest_fit = particle_pos[ibest], particle_fit[ibest]

        self.best_solution.iloc[:] = append(gbest_pos, gbest_fit)
        return gbest_pos, gbest_fit


if __name__ == '__main__':
    from benchmarks import *
    pso = ParticleSwarmOptimization(func=Eggholder(), debug=True)
    best_sol, best_val = pso.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(pso.eval_count)