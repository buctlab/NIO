from algorithms.algorithm import Algorithm, Ackley
from numpy import asarray, zeros, apply_along_axis, where, cumsum, concatenate, argsort, e, pi, sqrt, abs, max, min
from sklearn.cluster import KMeans
import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger('AMACO')
logger.setLevel('INFO')


class AMACO(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lower, self.upper = asarray(self.func.min_values), asarray(self.func.max_values)
        self.dim = self.func.dimension
        self.iter = 0

        index = pd.MultiIndex.from_product([list(range(self.iterations)), list(range(self.population))], names=['Iteration', 'Individual'])
        columns = list(range(self.dim))
        self.iter_swarm_pos = pd.DataFrame(index=index, columns=columns)

    def roulette_wheel_selection(self, weights):
        accumulation = cumsum(weights)
        p = self.Rand.rand() * accumulation[-1]
        chosen_index = -1
        for index in range(len(accumulation)):
            if accumulation[index] > p:
                chosen_index = index
        choice = chosen_index
        return choice

    def crowd(self, k, ant_pos, ant_fit):
        r"""

        :param k: Number of clusters
        :param ant_pos: Position of ant
        :param ant_fit: Fitness of ant
        :return:    c          f          0          1
                0   2  20.066109  -5.438046  14.439186
                1   1  20.911572 -32.760504 -12.954333
                ......
        """
        clf = KMeans(n_clusters=k)
        s = clf.fit(ant_pos)
        ant_class = pd.DataFrame(clf.labels_, columns=['c'])
        ant_pos = pd.DataFrame(ant_pos)
        ant_fit = pd.DataFrame(ant_fit, columns=['f'])
        result = pd.concat([ant_class, ant_fit, ant_pos], axis=1)
        return ant_pos, ant_fit, ant_class, result

    def boundary_revision(self, x):
        ir = where(x < self.lower)
        x[ir] = self.lower[ir]
        ir = where(x > self.upper)
        x[ir] = self.upper[ir]
        return x

    def construct_new_ant(self, result, FS_max, FS_min):
        new_ants = []
        new_ants_class = []
        new_ants_fitness = []
        for name, group in result.groupby('c'):
            FS_i_max = group.loc[group['f'].idxmax(), 'f']
            FS_i_min = group.loc[group['f'].idxmin(), 'f']
            temp = -(FS_i_max - FS_i_min) / (FS_max - FS_min + 10 ** (-31))
            sigma = 0.1 + 0.3 * e ** temp  # (5)
            group_sort = group.sort_values(by="f", ascending=False)
            W = []
            P = []
            temp1 = 1 / (sigma * len(group_sort) * sqrt(2 * pi))
            temp2 = 2 * (sigma ** 2) * (len(group_sort) ** 2)
            for i in range(len(group_sort)):  # 式(2)
                w = temp1 * e ** (- i ** 2 / temp2)
                W.append(w)
            for i in range(len(group_sort)):  # 式(1)
                p = W[i] / sum(W)
                P.append(p)
            temp3 = group['f'].idxmin()
            x_seed = result.iloc[temp3, 2:2 + self.dim]
            for i in range(len(group_sort)):
                j = self.roulette_wheel_selection(P)
                x_j = group_sort.iloc[j, 2:2 + self.dim]
                temp4 = self.Rand.rand()
                if temp4 <= 0.5:
                    mu = x_j
                else:
                    mu = x_j + self.Rand.rand() * (x_seed - x_j)
                add = 0
                for k in range(len(group_sort)):
                    add += abs(group.iloc[k, 2:2 + self.dim] - x_j)
                xi = self.Rand.uniform(0, 1)
                delta = xi * add / (len(group_sort))  # (4)
                ant = self.Rand.normal(loc=mu, scale=delta)
                ant = self.boundary_revision(ant)
                new_ants.append(ant)
                new_ants_class.append(name)
                new_ants_fitness.append(self.cost_function(ant))
        new_ants_class = pd.DataFrame(new_ants_class, columns=['c'])
        new_ants_fitness = pd.DataFrame(new_ants_fitness, columns=['f'])
        new_ants = pd.DataFrame(new_ants)
        new_result = pd.concat([new_ants_class, new_ants_fitness, new_ants], axis=1)
        return new_ants, new_ants_fitness, new_ants_class

    def replace(self, new_ants, new_ants_fitness, new_ants_class, ants, ants_fitness, ants_class):
        for index, row in new_ants.iterrows():
            distance_vector = []
            for i, r in ants.iterrows():
                distance_vector.append(sqrt((((row - r) ** 2).values).sum()))
            nearest = distance_vector.index(min(distance_vector))
            if new_ants_fitness.iloc[index]['f'] < ants_fitness.iloc[nearest]['f']:
                ants_fitness.iloc[nearest] = new_ants_fitness.iloc[index]
                ants.iloc[nearest] = row
                ants_class.iloc[nearest] = new_ants_class.iloc[index]
        result = pd.concat([ants_class, ants_fitness, ants], axis=1)
        return result

    def local_search(self, result):
        local_std_value = 10 ** (-4)
        N = 2
        Seed = pd.DataFrame()
        idmin = []
        for name, group in result.groupby('c'):
            event = group.loc[group['f'].idxmin()]
            idmin.append(group['f'].idxmin())
            Seed = Seed.append(event, ignore_index=True)
        FSE_min = Seed.loc[Seed['f'].idxmin(), 'f']
        FSE_max = Seed.loc[Seed['f'].idxmax(), 'f']
        flag = False
        if FSE_min <= 0:
            FSE_max = FSE_max + abs(FSE_min) + 10 ** (-31)
            flag = True
        Prob = []
        for index, row in Seed.iterrows():
            if flag:
                temp5 = row['f'] + abs(FSE_min) + 10 ** (-31)
                temp6 = FSE_max + abs(FSE_min) + 10 ** (-31)
                Prob.append(temp5 / temp6)
            else:
                Prob.append(row['f'] / FSE_max)
        # print(Prob)
        for index, row in Seed.iterrows():
            if self.Rand.rand() <= Prob[index]:
                for j in range(N):
                    LS = self.Rand.normal(loc=row[0:self.dim], scale=local_std_value)
                    LS = self.boundary_revision(LS)
                    if self.cost_function(LS) < row['f']:
                        result.loc[idmin[index], 'f'] = self.cost_function(LS)
                        result.iloc[idmin[index], 2:2 + self.dim] = LS
        return result

    def run(self):
        ant_pos = pd.DataFrame(self.initial_position())
        ant_fit = ant_pos.apply(self.cost_function, axis=1).astype('float64')

        result = None

        self.iter = 0
        while not self.stopping_criteria(self.iter):
            self.iter_swarm_pos.loc[self.iter] = ant_pos.values
            self.iter += 1

            FS_max, FS_min = max(ant_fit), min(ant_fit)
            # crowd
            K = 4  # 聚类个数
            ant_pos, ant_fit, ant_class, result = self.crowd(K, ant_pos, ant_fit)
            new_ants, new_ants_fitness, new_ants_class = self.construct_new_ant(result, FS_max, FS_min)
            result = self.replace(new_ants, new_ants_fitness, new_ants_class, ant_pos, ant_fit, ant_class)
            result = self.local_search(result)
        return result


if __name__ == '__main__':
    from benchmarks import Camel6, Crossit, Holdertable
    from benchmarks.cec2013 import CEC2013Convert
    iteration = 1000
    am_aco = AMACO(func=Camel6(), population=20, iterations=iteration, seed=1, debug=True)
    result = am_aco.run()
    logger.info("{res}".format(res=result))
