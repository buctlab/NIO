from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, apply_along_axis, where, argsort, arctan, pi, sin, tan, fabs, sqrt, sum, zeros, concatenate, append
from scipy.stats import levy
from scipy.special import gamma
import logging

logging.basicConfig()
logger = logging.getLogger('SSA')
logger.setLevel('INFO')


class SquirrelSearchAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gc = kwargs.pop('Gc', 1.9)  # gliding constant
        self.pdp = kwargs.pop('Pdp', 0.1)  # Predator presence probability
        self.rho = kwargs.pop('rho', 1.204)  # density of air
        self.v = kwargs.pop('V', 5.25)  # glide speed
        self.s = kwargs.pop('S', 154)  # surface area of body
        self.cd = kwargs.pop('CD', 0.6)  # the frictional drag coefficient
        self.hg = kwargs.pop('hg', 8)  # the loss in height occurred after gliding
        self.sf = kwargs.pop('sf', 18)  # scaling factor /in [16, 37] to make dg /in [0.5, 1.11]
        # self.sf = fabs(self.upper - self.lower)  # scaling factor

        self.sc = kwargs.pop('sc', 0.3)  # separate constant

    @staticmethod
    def sort_in_asc(loc, fit):
        r"""Sort the locations of flying squirrels in ascending order depending upon their fitness value

        :param loc: squirrels locations
        :param fit: squirrels fitness value
        :return: loc and fit sorted in ASC
        """
        indices = argsort(fit)
        return loc[indices], fit[indices]

    def squirrel_map_to_tree(self, loc):
        r"""Declare the flying squirrels on hickory nut tree, acorn nuts trees and normal trees.

        :param loc: Locations of flying squirrels
        :return: Locations of flying squirrels on: hickory nut tree & acorn nuts trees & normal trees
        """
        hickory_nut_tree = loc[0]
        acorn_nuts_trees = loc[1:4]
        normal_trees = loc[4:]
        return hickory_nut_tree, acorn_nuts_trees, normal_trees

    def random_separate(self, normal_loc):
        r"""Randomly select some flying squirrels which are on normal trees to
        move towards hickory nut tree and the remaining will move towards acorn nuts trees.

        :param normal_loc: Locations of flying squirrels on normal trees
        :return: Locations of flying squirrels move towards: hickory nut tree & acorn nuts trees
        """
        indices = self.Rand.permutation(normal_loc.shape[0])
        # TODO: unclear about how to separate flying squirrels on normal trees
        i = self.Rand.randint(1, normal_loc.shape[0])
        # i = round(self.sc * normal_loc.shape[0])
        normal_to_hickory_idx, normal_to_acorn_idx = indices[:i], indices[i:]
        return normal_loc[normal_to_hickory_idx], normal_loc[normal_to_acorn_idx]

    def gliding_distance(self):
        r"""generate flying squirrels gliding distance

        dg /in [9, 20]m
        dg is divided by a suitable non-zero value called as scaling factor (sf) obtained through rigorous experimentation on benchmark functions.
        sf /in [16, 37]
        dg/sf /in [0.5, 1.11] where sf = 18

        :return: dg divided by sf /in [0.5, 1.11]
        """
        cl = self.Rand.uniform(0.675, 1.5)  # lift coefficient
        l = 0.5 * self.rho * cl * self.v ** 2 * self.s  # lift
        d = 0.5 * self.rho * self.v ** 2 * self.s * self.cd  # frictional drag
        phi = arctan(d / l)  # glide angle
        dg = self.hg / tan(phi)  # gliding distance
        return dg / self.sf

    def random_location(self):
        r"""Random walk.

        :return: A random position of search space obeying uniform distribution
        """
        return self.Rand.uniform(asarray(self.lower), asarray(self.upper))

    def acorn_to_hickory(self, acorn_loc, hickory_loc):
        r"""Flying squirrels on acorn trees move towards hickory nut tree.

        :param acorn_loc: Location of flying squirrels on acorn trees
        :param hickory_loc: Location of flying squirrels on hickory nut tree
        :return: New location of flying squirrels on acorn trees
        """
        r1 = self.Rand.rand()
        if r1 >= self.pdp:
            dg = self.gliding_distance()
            # logger.info(dg)
            step = dg * self.gc * (hickory_loc - acorn_loc)
            return acorn_loc + step
        else:
            return self.random_location()

    def normal_to_acorn(self, normal_loc, acorn_loc):
        r"""Flying squirrels on normal trees move towards acorn trees.

        :param normal_loc: Location of flying squirrels on normal trees
        :param acorn_loc: Location of flying squirrels on acorn trees
        :return: New location of flying squirrels on normal trees moving towards acorn trees
        """
        r2 = self.Rand.rand()
        if r2 >= self.pdp:
            dg = self.gliding_distance()
            # logger.info(dg)
            step = dg * self.gc * (acorn_loc - normal_loc)
            return normal_loc + step
        else:
            return self.random_location()

    def normal_to_hickory(self, normal_loc, hickory_loc):
        r"""Flying squirrels on normal trees move towards hickory nut tree.

        :param normal_loc: Location of flying squirrels on normal trees
        :param hickory_loc: Location of flying squirrels on hickory nut tree
        :return: New location of flying squirrels on normal trees moving towards hickory nut tree.
        """
        r3 = self.Rand.rand()
        if r3 >= self.pdp:
            dg = self.gliding_distance()
            # logger.info(dg)
            step = dg * self.gc * (hickory_loc - normal_loc)
            return normal_loc + step
        else:
            return self.random_location()

    def seasonal_monitoring_condition(self, acorn_loc, hickory_loc, current_iteration, max_iteration):
        r"""Check whether if seasonal monitoring condition is satisfied or not.

        :param acorn_loc: Location of flying squirrels on acorn trees
        :param hickory_loc: Location of flying squirrels on hickory nut tree
        :param current_iteration: Current iteration
        :param max_iteration: Max iterations
        :return: True if seasonal constant (Sc) < the minimum value of seasonal constant (Smin)
        """
        sc = sqrt(sum((acorn_loc - hickory_loc) ** 2))  # seasonal constant
        t, tm = current_iteration, max_iteration
        s_min = 10e-6 / 365 ** (t * 2.5 / tm)
        return sc < s_min

    def levy_flight(self, beta=1.5):
        r"""The Levy flight defined in this algorithm.

        :param beta: A constant considered to be 1.5 in the present work
        :return: A Levy flight random value
        """
        sigma = ((gamma(1 + beta) * sin(pi * beta * 0.5)) / (gamma((1 + beta) * 0.5) * beta * 2 ** ((beta - 1) * 0.5)))\
                ** (1 / beta)
        ra, rb = self.Rand.rand(), self.Rand.rand()
        return 0.01 * ra * sigma / (fabs(rb) ** (1 / beta))

    def random_relocation(self):
        r"""Randomly relocate flying squirrels via Levy flight.

        :return: A random location of search space obeying levy flight
        """
        # TODO: It seems unclear about how to levy flight
        n = 1.5
        # return asarray(self.lower) + levy.pdf(n) * asarray(self.upper - self.lower)
        return asarray(self.lower) + self.levy_flight(n) * asarray(self.upper - self.lower)

    def run(self):
        # Init flying squirrels
        squirrel_location = self.initial_position()
        squirrel_fitness = apply_along_axis(self.cost_function, 1, squirrel_location)

        # Flying squirrels assigned to each kind of tree
        squirrel_location, squirrel_fitness = self.sort_in_asc(squirrel_location, squirrel_fitness)
        hickory_nut_tree_loc, acorn_nuts_trees_loc, normal_trees_loc = self.squirrel_map_to_tree(squirrel_location)
        normal_to_hickory, normal_to_acorn = self.random_separate(normal_trees_loc)

        # Global best
        best_sol, best_val = squirrel_location[0], squirrel_fitness[0]

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = squirrel_location
            self.iter_solution.loc[self.iter] = append(best_sol, best_val)
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=self.iter_solution.loc[
                                                                                 self.iter].to_dict()))

            # Generate new locations
            acorn_nuts_trees_loc = asarray([self.acorn_to_hickory(acorn_nuts_trees_loc[i], hickory_nut_tree_loc) for i in range(acorn_nuts_trees_loc.shape[0])])
            normal_to_hickory = asarray([self.normal_to_hickory(normal_to_hickory[i], hickory_nut_tree_loc) for i in range(normal_to_hickory.shape[0])])
            # TODO: acorn_nuts_trees_loc[i % 3] ?
            normal_to_acorn = asarray([self.normal_to_acorn(normal_to_acorn[i], acorn_nuts_trees_loc[i % 3]) for i in range(normal_to_acorn.shape[0])])

            squirrel_location = concatenate(([hickory_nut_tree_loc], acorn_nuts_trees_loc, normal_to_hickory, normal_to_acorn), axis=0)
            squirrel_location = apply_along_axis(self.boundary_handle, 1, squirrel_location)
            squirrel_fitness = apply_along_axis(self.cost_function, 1, squirrel_location)
            squirrel_location, squirrel_fitness = self.sort_in_asc(squirrel_location, squirrel_fitness)

            # Random relocation at the end of winter season
            flag = asarray([self.seasonal_monitoring_condition(acorn_nuts_trees_loc[i], hickory_nut_tree_loc, self.iter, self.iterations) for i in range(acorn_nuts_trees_loc.shape[0])])
            if flag.all():
                # logger.info("{iter}/{iteration}-Winter is coming!".format(iter=self.iter, iteration=self.iterations))
                # Replace flying squirrels on normal trees
                # TODO: It seems unclear about what is flying squirrels going to dead
                squirrel_location[4:] = asarray([self.random_relocation() for i in range(self.population-4)])
                squirrel_fitness = apply_along_axis(self.cost_function, 1, squirrel_location)

            hickory_nut_tree_loc, acorn_nuts_trees_loc, normal_trees_loc = self.squirrel_map_to_tree(squirrel_location)
            normal_to_hickory, normal_to_acorn = self.random_separate(normal_trees_loc)

            # Update global best
            if squirrel_fitness[0] < best_val:
                best_sol, best_val = squirrel_location[0], squirrel_fitness[0]

        self.best_solution.iloc[:] = append(best_sol, best_val)
        return best_sol, best_val


if __name__ == '__main__':
    from benchmarks import *
    ssa = SquirrelSearchAlgorithm( debug=True)
    best_sol, best_val = ssa.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
    print(ssa.eval_count)