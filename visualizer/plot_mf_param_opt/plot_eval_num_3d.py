# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from numpy import arange
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotEvalNum3D')
logger.setLevel('INFO')


class PlotEvalNum3D:

    def __init__(self, data, path, show=False):
        self.data = data
        self.path = path
        self.show_flag = show
        (filepath, tempfilename) = os.path.split(path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        (filename, extension) = os.path.splitext(tempfilename)
        self.format = extension[1:]

    def plot_eval(self):
        data = self.data
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['tab:brown', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange']
        # colors = ['sandybrown', 'lightgreen', 'salmon', 'mediumpurple', 'khaki']
        yticks = [0, 1, 2, 3, 4]
        for c, k in zip(reversed(colors), reversed(yticks)):
            xs = data.index
            ys = data[data.columns[k]]
            cs = [c] * len(xs)

            ax.bar(left=xs, height=ys, zs=k, zdir='y', color=cs, alpha=0.8)
            # ax.bar(left=xs, height=ys, zs=k, zdir='y', color=cs)

        ax.set_xlabel('Optimized-NIO', labelpad=10, fontsize=14)
        ax.set_ylabel('FCF', rotation=45, labelpad=10, fontsize=14)
        ax.set_zlabel('Eval Counts', labelpad=10, fontsize=14)

        ax.set_yticks(yticks)
        ax.set_yticklabels(data.columns)
        # ax.set_title("Optimizer GWO")
        plt.savefig(self.path, format=self.format, dpi=80)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    mf_func = ['Power', 'ReLU', 'Sigmoid', 'Sin', 'Fixed']
    optimized = ['cs', 'de', 'kh', 'pso', 'ssa', 'wwo']
    data = pd.DataFrame(index=optimized, columns=mf_func)
    for func in mf_func:
        if func == 'Fixed':
            csv_path = r"../../tests/mf_param_opt_tests/output/OriginalOptimizerGWOTest.csv"
        else:
            csv_path = r"../../tests/mf_param_opt_tests/output/MF{func}OptimizerGWOTest.csv".format(func=func)
        df = pd.read_csv(csv_path, header=0, index_col=0)
        data[func] = df['EvalCounts'].values
    logger.info(data)
    box_3d_path = r"output/PlotEvalNum3D/MFOptimizerGWO.png"

    pen = PlotEvalNum3D(data=data, path=box_3d_path, show=True)
    pen.plot_eval()
