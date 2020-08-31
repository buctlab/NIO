import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBox')
logger.setLevel('INFO')


class PlotBox:

    def __init__(self, data, benchmark, path, meta, optimized, show=False):
        # plt.style.use("ggplot")

        self.data = data  # DataFrame
        self.benchmark = benchmark

        self.path = path
        self.show_flag = show
        self.meta= meta
        self.optimized = optimized
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_origin(self):
        data = self.data
        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()
        ax.boxplot(x=data.values, labels=data.columns, whis=1.5)
        ax.tick_params(labelsize=16)
        ax.set_xticklabels(labels=data.columns, fontsize=25)
        ax.set_ylabel("CFV", fontsize=25)
        ax.set_title('Meta-NIO: {meta},     Optimized-NIO: {optimized}'.format(meta=self.meta, optimized=self.optimized), fontsize=25)
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def plot_error(self):
        standard_value = self.benchmark.get_optimum()[-1]
        data = (self.data - standard_value).abs()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()

        ax.boxplot(x=data.values, labels=data.columns, whis=1.5)
        ax.tick_params(labelsize=16)
        ax.set_xticklabels(labels=data.columns, fontsize=25)
        ax.set_ylabel("CFV Error", fontsize=25)
        ax.set_title(
            'Meta-NIO: {meta},     Optimized-NIO: {optimized}'.format(meta=self.meta, optimized=self.optimized),
            fontsize=25)
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def plot_log(self):
        standard_value = self.benchmark.get_optimum()[-1]
        data = (self.data - standard_value).abs()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()

        ax.set_yscale('log')
        ax.boxplot(x=data.values, labels=data.columns, whis=1.5)
        ax.tick_params(labelsize=16)
        ax.set_xticklabels(labels=data.columns, fontsize=25)
        ax.set_ylabel("Log(CFV Error)", fontsize=25)
        # ax.set_title('Box', fontsize=25)
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Michalewicz
    csv_path = r"../../tests/parameter_optimization_tests/parameter_evaluation_tests/output/MultiRunsTest/TestOptimizerCS/DE.csv"
    box_path = r"output/PlotBox/DE.png"
    multi_runs_data = pd.read_csv(csv_path, header=0, index_col=0)
    multi_runs_data.drop(['mean', 'std', 'median', 'best', 'worst'], inplace=True)

    pb = PlotBox(data=multi_runs_data, benchmark=Michalewicz(dimension=5), path=box_path,meta="CS", optimized="CS", show=True)
    # pb.plot_origin()
    pb.plot_error()
    # pb.plot_log()
