
from matplotlib import pyplot as plt
from numpy import linspace, meshgrid, asarray
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotSwarmPos2D')
logger.setLevel('INFO')


class PlotSwarmPos2D:

    def __init__(self, data, benchmark, path, show=False):
        self.data = pd.DataFrame(data)
        self.benchmark = benchmark
        self.lower, self.upper = self.benchmark.lower, self.benchmark.upper
        self.ref_points = asarray(self.benchmark.get_optimum()[0])
        self.show_flag = show
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_swarm_pos(self, i, algorithm, filename=None):
        r"""Plot swarm pos with contour at i-th generation.

        :param i: Current generation.
        :param algorithm: Algorithm name show in title.
        :return: None.
        """
        data = self.data.loc[i]
        n = 256
        x = linspace(self.lower[0], self.upper[0], n)
        y = linspace(self.lower[1], self.upper[1], n)
        X, Y = meshgrid(x, y)
        Z = self.benchmark.eval([X, Y])

        fig = plt.figure(figsize=(6, 6))
        ax = fig.subplots()
        ax.contour(X, Y, Z, 6, alpha=0.75, cmap=plt.get_cmap('Paired'))

        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c='black', marker='o', label="Individual Position")

        if self.ref_points.shape[1] > 0:
            ax.scatter(self.ref_points[:, 0], self.ref_points[:, 1], c='r', s=100, marker='*', label="Global Minimum")
        ax.legend(loc='upper right')

        if self.benchmark is not None:
            # The boundary point causes the side to be white. Set the fixed coordinate display range.
            ax.axis([self.lower[0], self.upper[0], self.lower[1], self.upper[1]])
        ax.set_title('{alg} {n}-Dim {benchmark} Gen.{i}'.format(alg=algorithm, n=self.benchmark.dimension,
                                                                benchmark=self.benchmark.__class__.__name__, i=i),
                     fontsize=20)
        if filename:
            plt.savefig(r"{path}/{file}.eps".format(path=self.path, file=filename), dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Ackley, Michalewicz
    csv_path = r"../tests/algorithm_parallel_tests/output/SwarmPosTest/2D/AntLionOptimizer_Ackley.csv"
    swarm_pos_path = r"output/PlotSwarmPos2D"
    fig_file = "S(q)SA_Michalewicz"
    swarm_pos_data = pd.read_csv(csv_path, header=0, index_col=[1, 1])

    pc = PlotSwarmPos2D(data=swarm_pos_data, benchmark=Michalewicz(), path=swarm_pos_path)
    pc.plot_swarm_pos(i=10, algorithm='S(q)SA', filename=fig_file)
