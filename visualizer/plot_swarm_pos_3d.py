from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, meshgrid, array
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotSwarmPos3D')
logger.setLevel('INFO')


class PlotSwarmPos3D:

    def __init__(self, data, benchmark, path, show=False):
        self.data = pd.DataFrame(data)
        self.benchmark = benchmark
        self.lower, self.upper = self.benchmark.lower, self.benchmark.upper
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

        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)

        scale = 0.05
        X = arange(self.lower[0], self.upper[0], scale)
        Y = arange(self.lower[1], self.upper[1], scale)
        X, Y = meshgrid(X, Y)
        Z = self.benchmark.eval(array([X, Y]))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3, cmap='rainbow')

        data['Fitness'] = data.apply(self.benchmark.eval, axis=1)
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data['Fitness'], c='black', marker='o')

        # The boundary point causes the side to be white. Set the fixed coordinate display range.
        ax.axis([self.lower[0], self.upper[0], self.lower[1], self.upper[1]])
        ax.set_title('{alg} {n}-Dim {benchmark} Gen.{i}'.format(alg=algorithm, n=self.benchmark.dimension,
                                                                benchmark=self.benchmark.__class__.__name__, i=i),
                     fontsize=20)
        if filename:
            plt.savefig(r"{path}/{file}".format(path=self.path, file=filename), dpi=200, bbox_inches='tight')
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Ackley, Michalewicz
    csv_path = r"../tests/algorithm_parallel_tests/output/SwarmPosTest/2D/AntLionOptimizer_Michalewicz.csv"
    swarm_pos_path = r"output/PlotSwarmPos3D"
    fig_file = "ALO_Michalewicz.png"
    swarm_pos_data = pd.read_csv(csv_path, header=0, index_col=[0, 1])

    pc = PlotSwarmPos3D(data=swarm_pos_data, benchmark=Michalewicz(), path=swarm_pos_path, show=True)
    pc.plot_swarm_pos(i=10, algorithm='ALO', filename=fig_file)
