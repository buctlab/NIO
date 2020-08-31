from matplotlib import pyplot as plt
from numpy import full
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotConvergence')
logger.setLevel('INFO')


class PlotConvergence:

    def __init__(self, data, benchmark, path, show=False):
        self.line_styles = ['-', '--', '-.', ':']
        style_list = ['default', 'classic'] + sorted(
            style for style in plt.style.available if style != 'classic')
        plt.style.use(style_list[0])

        self.data = data  # DataFrame
        self.benchmark = benchmark
        self.baseline_val = self.benchmark.get_optimum()[-1]

        self.path = path
        self.show_flag = show
        (filepath, tempfilename) = os.path.split(path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        (filename, extension) = os.path.splitext(tempfilename)
        self.format = extension[1:]

    def plot_convergence(self):
        data = self.data

        fig = plt.figure(figsize=(10, 6))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_xlabel("Gen", fontsize=16)
        ax.set_ylabel("CFV", fontsize=16)

        X = data.index
        lines = data.columns

        for line in lines:
            Y = data[line]
            ax.plot(X, Y, linestyle='-', label=line)

        # base line
        Y = full(len(X), self.baseline_val)
        ax.plot(X, Y, color='k', linestyle=self.line_styles[3], linewidth=2, label='Base Line')
        # ax.set_title('Convergence of 10-Level Fidelity', fontsize=16)
        # ax.tick_params(labelsize=12)
        ax.legend(loc='upper right')  # show legend
        plt.savefig(self.path, format=self.format, dpi=80)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Sphere

    csv_path = r"../../../tests/mf_param_opt_tests/parameter_evaluation_tests/output/ConvergenceTest/Sphere50/FidelityLevelTest/FidelityLevelConvergence.csv"
    convergence_fig_path = r"output/tt/FidelityLevelConvergence.png"
    convergence_data = pd.read_csv(csv_path, header=0, index_col=0)

    pc = PlotConvergence(data=convergence_data, benchmark=Sphere(dimension=50), path=convergence_fig_path, show=True)
    pc.plot_convergence()
