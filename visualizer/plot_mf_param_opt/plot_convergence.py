from matplotlib import pyplot as plt
from matplotlib import colors
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

        fig = plt.figure(figsize=(6, 6))
        # left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        # ax = fig.add_axes([left, bottom, width, height])
        ax = fig.add_subplot(111)
        ax.set_xlabel("Gen", fontsize=16)
        ax.set_ylabel("CFV", fontsize=16)

        X = data.index
        lines = data.columns  # ['Raw', 'Non-MultiFidelity', 'ReLU-Fidelity', 'Sigmoid-Fidelity', 'Sin-Fidelity', 'Power-Fidelity']
        colors = ['black', 'black', 'green', 'cyan', 'blue', 'violet', 'magenta', 'red', 'orange', 'yellow']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        colors = ['black', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

        for (color, line) in zip(colors, lines):
            Y = data[line]
            if line == 'Recommended':
                ax.plot(X, Y, color=color, linestyle=':', linewidth=2, label=line)
            else:
                ax.plot(X, Y, color=color, linestyle='-', label=line)

        # base line
        # Y = full(len(X), self.baseline_val)
        # ax.plot(X, Y, color='r', linestyle=self.linestyles[3], linewidth=3, label='Base Line')
        # ax.set_title('Convergence Curve', fontsize=25)
        ax.legend(loc='upper right', prop={'size': 14})  # show legend
        plt.savefig(self.path, format=self.format, dpi=80, bbox_inches='tight')
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Sphere

    csv_path = r"../../tests/mf_param_opt_tests/parameter_evaluation_tests/output/ConvergenceTest/MFOptimizerCS/CS.csv"
    convergence_fig_path = r"output/PlotConvergence/CS.png"
    convergence_data = pd.read_csv(csv_path, header=0, index_col=0)

    pc = PlotConvergence(data=convergence_data, benchmark=Sphere(dimension=50), path=convergence_fig_path, show=True)
    pc.plot_convergence()
