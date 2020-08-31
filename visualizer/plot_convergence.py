import matplotlib


from matplotlib import pyplot as plt
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotConvergence')
logger.setLevel('INFO')

color = ["cadetblue", "darksalmon", "indigo", "palegreen", "bisque", "steelblue", "cyan"]
line = ['+','v','*','>','^','_','2','3','4']


class PlotConvergence:

    def __init__(self, data, benchmark, path, show=False):
        # self.colors = list(colors.CSS4_COLORS)
        plt.style.use("default")
        self.linestyles = ['-', '--', '-.', ':']

        self.data = pd.DataFrame()
        self.data = data
        self.benchmark = benchmark
        self.baseline_val = self.benchmark.get_optimum()[-1]

        self.path = path
        self.show_flag = show
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_convergence(self):
        data = self.data

        fig = plt.figure(figsize=(10, 6))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_xlabel("Gen", fontsize=25)
        ax.set_ylabel("CFV", fontsize=25)

        X = data.index
        algorithms = data.columns

        # plot each convergence line
        for i in range(len(algorithms)):
            Y = data[algorithms[i]]
            # ax.plot(X, Y, color=self.colors[i * 3 + 2], linestyle='solid', label=algorithms[i])
            if i < 10:
                ax.plot(X, Y, linestyle='solid', label=algorithms[i])
            else:
                ax.plot(X, Y, color=color[i - 10], label=algorithms[i])

        # base line
        Y = [self.baseline_val] * len(X)
        ax.plot(X, Y, color='r', linestyle=self.linestyles[3], linewidth=3, label='Base Line')
        ax.set_title(
            '{n}D - {benchmark}'.format(n=self.benchmark.dimension, benchmark=self.benchmark.__class__.__name__),
            fontsize=25)
        ax.legend(loc='upper right')  # show legend
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def plot_convergence_with_subfig(self):
        data = self.data

        fig = plt.figure()
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        main_ax = fig.add_axes([left, bottom, width, height])
        main_ax.set_xlabel("Iteration")
        main_ax.set_ylabel("Convergence Value")

        X = data.index
        algorithms = data.columns

        # plot each convergence line
        for i in range(len(algorithms)):
            Y = data[algorithms[i]]
            main_ax.plot(X, Y, color=self.colors[i * 3 + 2], linestyle=line[i], label=algorithms[i])

        # plot sub figure
        left, bottom, width, height = 0.3, 0.4, 0.25, 0.4
        sub_ax = fig.add_axes([left, bottom, width, height])

        sub_X = data.index[5:20]
        for i in range(len(algorithms)):
            sub_Y = data.loc[6:20, algorithms[i]]
            sub_ax.plot(sub_X, sub_Y, color=self.colors[i * 3 + 2], linestyle='solid', label=algorithms[i])

        # base line
        Y = [self.baseline_val] * len(X)
        main_ax.plot(X, Y, color='r', linestyle=self.linestyles[3], linewidth=3, label='Base Line')
        main_ax.set_title('Convergence of Running {n}-Dim {benchmark}'.format(n=self.benchmark.dimension,
                                                                              benchmark=self.benchmark.__class__.__name__))
        main_ax.legend()  # show legend
        if self.path is not None:
            plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import *

    csv_path = r"../tests/algorithm_parallel_tests/output/AllTests2019-12/convergence/30D/Stybtang.csv"
    convergence_fig_path = r"output/PlotConvergence/case3.eps"
    convergence_data = pd.read_csv(csv_path, header=0, index_col=0)
    data = pd.DataFrame()
    # data = convergence_data
    data['A'] = convergence_data['FireflyAlgorithm']
    data['B'] = convergence_data['AntLionOptimizer']
    print(data)

    pc = PlotConvergence(data=data, benchmark=Stybtang(dimension=30), path=convergence_fig_path,
                         show=True)
    pc.plot_convergence()
