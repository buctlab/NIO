import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('Plot1stDimTrajectory')
logger.setLevel('INFO')


class Plot1stDimTrajectory:

    def __init__(self, data, benchmark, path, show=False):
        self.data = data  # DataFrame
        self.benchmark = benchmark
        self.baseline_val = self.benchmark.get_optimum()[-1]

        self.path = path
        self.show_flag = show
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_trajectory(self):
        data = self.data

        fig = plt.figure(figsize=(10, 6))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_xlabel("Gen", fontsize=25)
        ax.set_ylabel("Error", fontsize=25)

        X = data.index
        lines = data.columns  # ['Before', 'After']

        # Before
        Y1 = data[lines[0]]
        ax.plot(X, Y1, color='b', linestyle=':', label=lines[0])
        # After
        Y2 = data[lines[1]]
        ax.plot(X, Y2, color='r', linestyle='-', label=lines[1])

        ax.set_title('Trajectory in 1st dimension', fontsize=25)
        ax.legend(loc='upper right')  # show legend
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Michalewicz

    csv_path = r"../../tests/parameter_optimization_tests/parameter_evaluation_tests/output/TrajectoryTest/TestOptimizerCS/CS.csv"
    trajectory_fig_path = r"output/Plot1stDimTrajectory/CS.png"
    trajectory_data = pd.read_csv(csv_path, header=0, index_col=0)

    pt = Plot1stDimTrajectory(data=trajectory_data, benchmark=Michalewicz(dimension=5), path=trajectory_fig_path, show=True)
    pt.plot_trajectory()
