from algorithms import SquirrelSearchAlgorithm
from numpy import apply_along_axis, empty_like, asarray, concatenate, sum, zeros, argmax, argmin, where, max, sqrt
from sklearn.cluster import KMeans
import logging

logging.basicConfig()
logger = logging.getLogger('MultiModalSSA')
logger.setLevel('INFO')


class MultiModalSSA(SquirrelSearchAlgorithm):

    def __init__(self, **kwargs):
        self.min_cluster_size = kwargs.pop('cluster_size', 20)  # >6
        self.cluster_num = kwargs.pop('cluster_num', 4)
        self.population = kwargs.setdefault('population', 20*self.cluster_num)
        super().__init__(**kwargs)

    def cluster(self, loc, k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(loc)
        cluster, center = kmeans.labels_, kmeans.cluster_centers_
        loc, cluster = self.cluster_equilibrium(loc, cluster, center, k)
        return loc, cluster, center

    def cluster_equilibrium(self, loc, cluster, center, k):
        count = zeros(k)
        for i in range(k):
            count[i] = sum(cluster == i)
        vacancy = self.min_cluster_size - count  # 补缺个数
        for i in range(k):
            if vacancy[i] > 0:
                for j in range(int(vacancy[i])):
                    rc = argmax(count)  # replace cluster
                    count[rc] -= 1
                    vacancy[rc] += 1

                    idx = where(cluster == rc)
                    ri = self.Rand.choice(idx[0], 1)  # replace index by random selection
                    cluster[ri] = i
                    loc[ri] = self.Rand.normal(loc=center[i], scale=0.01)
        return loc, cluster

    def replace(self, original_loc, original_fit, new_loc, new_fit):
        loc, fit = original_loc.copy(), original_fit.copy()
        for i in range(self.population):
            distance = sqrt(sum((original_loc - new_loc[i]) ** 2, axis=1))
            nearest = argmin(distance)
            if new_fit[i] < original_fit[nearest]:
                loc[nearest] = new_loc[i]
                fit[nearest] = new_fit[i]
        return loc, fit

    def generate_new_solution(self, squirrel_location):
        hickory_nut_tree_loc, acorn_nuts_trees_loc, normal_trees_loc = self.squirrel_map_to_tree(squirrel_location)
        normal_to_hickory, normal_to_acorn = self.random_separate(normal_trees_loc)

        # Generate new locations
        acorn_nuts_trees_loc = asarray([self.acorn_to_hickory(acorn_nuts_trees_loc[i], hickory_nut_tree_loc) for i in
                                        range(acorn_nuts_trees_loc.shape[0])])
        normal_to_hickory = asarray([self.normal_to_hickory(normal_to_hickory[i], hickory_nut_tree_loc) for i in
                                     range(normal_to_hickory.shape[0])])
        normal_to_acorn = asarray([self.normal_to_acorn(normal_to_acorn[i], acorn_nuts_trees_loc[i % 3]) for i in
                                   range(normal_to_acorn.shape[0])])

        squirrel_location = concatenate(
            ([hickory_nut_tree_loc], acorn_nuts_trees_loc, normal_to_hickory, normal_to_acorn), axis=0)
        squirrel_location = apply_along_axis(self.boundary_handle, 1, squirrel_location)
        squirrel_fitness = apply_along_axis(self.cost_function, 1, squirrel_location)
        squirrel_location, squirrel_fitness = self.sort_in_asc(squirrel_location, squirrel_fitness)

        # Random relocation at the end of winter season
        flag = asarray([self.seasonal_monitoring_condition(acorn_nuts_trees_loc[i], hickory_nut_tree_loc, self.iter,
                                                           self.iterations) for i in
                        range(acorn_nuts_trees_loc.shape[0])])
        if flag.all():
            # Replace flying squirrels on normal trees
            squirrel_location[4:] = asarray([self.random_relocation() for i in range(self.population - 4)])
        return squirrel_location

    def run(self):
        # Init flying squirrels
        squirrel_location = self.initial_position()
        squirrel_location, cluster, center = self.cluster(squirrel_location, self.cluster_num)
        squirrel_fitness = apply_along_axis(self.cost_function, 1, squirrel_location)

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter += 1
            self.iter_swarm_pos.loc[self.iter] = squirrel_location
            if self.debug:
                logger.info("Iteration:{i}/{iterations} - {iter_sol}".format(i=self.iter, iterations=self.iterations,
                                                                             iter_sol=squirrel_fitness))
            new_loc = empty_like(squirrel_location)
            for i in range(self.cluster_num):
                idx = where(cluster == i)
                loc = squirrel_location[idx]
                new_loc[idx] = self.generate_new_solution(loc)
            new_fit = apply_along_axis(self.cost_function, 1, new_loc)

            squirrel_location, squirrel_fitness = self.replace(squirrel_location, squirrel_fitness, new_loc, new_fit)
            squirrel_location, squirrel_fitness = self.sort_in_asc(squirrel_location, squirrel_fitness)
            squirrel_location, cluster, center = self.cluster(squirrel_location, self.cluster_num)

        return squirrel_location, squirrel_fitness


if __name__ == '__main__':
    from benchmarks import Camel6
    from visualizer.animation import PlotSwarmAnimation
    benchmark = Camel6(dimension=2)
    mm_ssa = MultiModalSSA(func=benchmark, iterations=300, cluster_num=2, cluster_size=10, population=60, debug=True)
    mm_ssa.run()
    swarm_pos = mm_ssa.iter_swarm_pos
    swarm_animation_path = r"output/PlotSwarmAnimation/MMSSA_Camel6.mp4"
    psa = PlotSwarmAnimation(swarm_pos, Camel6(), swarm_animation_path)
    psa.plot()
    psa.save()
