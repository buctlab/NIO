from algorithms.algorithm import Algorithm, Ackley
from scipy.spatial.distance import euclidean as ed
from numpy import apply_along_axis, argmin, asarray, sqrt, where, append
import logging

logging.basicConfig()
logger = logging.getLogger('FOA')
logger.setLevel('INFO')


class FruitFly(Algorithm):

    def __init__(self, **kwargs):
        # population = {3, 6, 9}
        super().__init__(**kwargs)

    def initial_position_trans(self):
        x_axis = self.initial_position()  # the position of X axis
        y_axis = self.initial_position()
        return x_axis, y_axis

    @staticmethod
    def distance(x_axis, y_axis):
        return sqrt(x_axis ** 2 + y_axis ** 2)

    @staticmethod
    def get_solution(dist):
        return 1 / dist

    def new_location(self, axis):
        return axis + self.Rand.uniform(self.lower, self.upper, self.dim)

    def boundary_handle(self, x):
        ir = where(x > self.upper)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        ir = where(x < self.lower)
        x[ir] = self.Rand.uniform(self.lower[ir], self.upper[ir])
        return x

    def run(self):
        x_axis, y_axis = self.initial_position_trans()  # the position of X, Y axis
        dist = self.distance(x_axis, y_axis)  # Calculate the distance
        sol = self.get_solution(dist)  # the solution set
        sol = apply_along_axis(self.boundary_handle, 1, sol)
        fitness = apply_along_axis(self.cost_function, 1, sol)

        i_best = argmin(fitness)  # Get the index of min fitness
        best_smell = fitness[i_best]
        best_x_axis = x_axis[i_best]  # the X axis of min fitness
        best_y_axis = y_axis[i_best]  # the Y axis of min fitness
        global_smell_best = best_smell
        global_best_sol = sol[i_best]

        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = sol
            self.iter_solution.loc[self.iter] = append(global_best_sol, global_smell_best)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            # Refer to the process of initializing
            x_axis = asarray([self.new_location(best_x_axis) for i in range(self.population)])
            y_axis = asarray([self.new_location(best_y_axis) for i in range(self.population)])
            dist = self.distance(x_axis, y_axis)
            sol = self.get_solution(dist)
            sol = apply_along_axis(self.boundary_handle, 1, sol)
            fitness = apply_along_axis(self.cost_function, 1, sol)

            i_best = argmin(fitness)
            best_smell = fitness[i_best]
            # If the new value is smaller than the best value,update the best value
            if best_smell < global_smell_best:
                best_x_axis = x_axis[i_best]
                best_y_axis = y_axis[i_best]
                global_best_sol = sol[i_best]
                global_smell_best = best_smell

        self.best_solution.iloc[:] = append(global_best_sol, global_smell_best)
        return global_best_sol, global_smell_best


if __name__ == '__main__':
    from benchmarks import *
    foa = FruitFly(func=Michalewicz(dimension=30), iterations=2, debug=True)
    best_sol, best_val = foa.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(foa.eval_count)
