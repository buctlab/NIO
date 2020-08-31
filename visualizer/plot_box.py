
from matplotlib import pyplot as plt
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBox')
logger.setLevel('INFO')


class PlotBox:

    def __init__(self, data, benchmark, path, show=False, rotate=True):
        # plt.style.use("ggplot")
        self.data = pd.DataFrame()
        self.data = data
        self.benchmark = benchmark
        self.path = path
        self.show_flag = show
        self.rotate_flag = rotate
        self.showbox = [False, False, False, True, False]
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_origin(self):
        data = self.data
        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()
        ax.boxplot(x=data.values, labels=data.columns, whis=1.5,showbox=self.showbox)
        ax.tick_params(labelsize=8, rotation=45*self.rotate_flag)
        ax.set_xticklabels(labels=data.columns, fontsize=8)
        ax.set_ylabel("CFV", fontsize=25)
        ax.set_title(
            'Multiple Tests'.format(n=self.benchmark.dimension, benchmark=self.benchmark.__class__.__name__),
            fontsize=25)
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
        # ax.tick_params(labelsize=8, rotation=45 * self.rotate_flag)
        ax.set_xticklabels(labels=data.columns, fontsize=8, rotation=45*self.rotate_flag)
        ax.set_ylabel("CFV Error", fontsize=25)
        ax.set_title(
            '{n}-Dim {benchmark}'.format(n=self.benchmark.dimension, benchmark=self.benchmark.__class__.__name__),
            fontsize=25)
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def plot_log(self):
        # data = self.data.abs()
        standard_value = self.benchmark.get_optimum()[-1]
        data = (self.data - standard_value).abs()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()

        ax.set_yscale('log')
        ax.boxplot(x=data.values, labels=data.columns, whis=1.5)
        # ax.tick_params(labelsize=8, rotation=45*self.rotate_flag)
        ax.set_xticklabels(labels=data.columns, fontsize=8, rotation=45*self.rotate_flag)
        ax.set_ylabel("Log(CFV Error)", fontsize=25)
        ax.set_title(
            '{n}-Dim {benchmark}'.format(n=self.benchmark.dimension, benchmark=self.benchmark.__class__.__name__),
            fontsize=25)
        plt.savefig(self.path, dpi=200)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Ackley
    csv_path = r"mtt.csv"
    box_path = r"output/PlotBox/Ackley.png"
    multi_runs_data = pd.read_csv(csv_path, header=0, index_col=0)

    pb = PlotBox(data=multi_runs_data, benchmark=Ackley(), path=box_path)
    pb.plot_origin()
    # pb.plot_error()
    # pb.plot_log()
