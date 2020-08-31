import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
from numpy import full
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotConvergence')
logger.setLevel('INFO')


class PlotConvergence:
    def __init__(self, data, benchmark, path,  meta, optimized, show=False):
        self.colors = list(colors.CSS4_COLORS)
        self.linestyles = ['-', '--', '-.', ':']

        self.data = data  # DataFrame
        self.benchmark = benchmark
        self.baseline_val = self.benchmark.get_optimum()[-1]

        self.path = path
        self.show_flag = show
        self.meta = meta
        self.optimized= optimized
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_convergence(self):
        data = self.data

        fig = plt.figure(figsize=(10, 6))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_title("Meta-NIO: {},     Optimized-NIO: {}".format(self.meta, self.optimized),fontsize=25)
        ax.set_xlabel("Gen", fontsize=25)
        ax.set_ylabel("CFV", fontsize=25)

        X = data.index
        lines = data.columns  # ['Before', 'After'] => Raw, Optimized

        # Before
        Y1 = data[lines[0]]
        ax.plot(X, Y1, color='b', linestyle=':', label=lines[0])
        # After
        Y2 = data[lines[1]]
        ax.plot(X, Y2, color='r', linestyle='-', label=lines[1])

        # base line
        # Y = full(len(X), self.baseline_val)
        # ax.plot(X, Y, color='r', linestyle=self.linestyles[3], linewidth=3, label='Base Line')
        # ax.set_title('Convergence Curve', fontsize=25)
        ax.legend(loc='upper right')  # show legend
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Michalewicz

    csv_path = r"../../tests/parameter_optimization_tests/parameter_evaluation_tests/output/ConvergenceTest/TestOptimizerCS/CS.csv"
    convergence_fig_path = r"output/PlotConvergence/CS.eps"
    convergence_data = pd.read_csv(csv_path, header=0, index_col=0)

    pc = PlotConvergence(data=convergence_data, benchmark=Michalewicz(dimension=5), path=convergence_fig_path,meta="CS", optimized="CS", show=True)
    pc.plot_convergence()
