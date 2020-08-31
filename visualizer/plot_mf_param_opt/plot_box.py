from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBox')
logger.setLevel('INFO')


class PlotBox:

    def __init__(self, data, benchmark, path, show=False):
        self.data = data  # DataFrame
        self.benchmark = benchmark

        self.path = path
        self.show_flag = show
        (filepath, tempfilename) = os.path.split(path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        (filename, extension) = os.path.splitext(tempfilename)
        self.format = extension[1:]

    def plot_seaborn(self):
        data = self.data
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, width=0.5)
        # sns.swarmplot(data=data)
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(labels=data.columns, fontsize=14)
        ax.set_ylabel("CFV", fontsize=16)
        plt.savefig(self.path, format=self.format, dpi=80)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def plot_edge(self):
        data = self.data
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        colors = ['black', 'blue', 'green', 'red', 'purple', 'brown']
        for i in range(data.shape[1]):
            c = colors[i]
            ax.boxplot(x=data.ix[:, i].values,
                       positions=[i+1],
                       labels=[data.columns[i]],
                       widths=0.5,
                       sym='+',
                       # patch_artist=True,  # fill with color
                       # boxprops=dict(facecolor=c, color=c),
                       boxprops=dict(color=c),
                       capprops=dict(color=c),
                       whiskerprops=dict(color=c),
                       flierprops=dict(color=c, markeredgecolor=c),
                       medianprops=dict(color=c)
                       )

        ax.set_xlim(0.5, data.shape[1]+0.5)
        ax.tick_params(labelsize=12)
        plt.xticks(list(range(1, data.shape[1]+1)), data.columns, fontsize=14)
        # ax.set_xticklabels(labels=data.columns, fontsize=14)
        ax.set_ylabel("CFV", fontsize=16)
        # ax.set_title('Box', fontsize=25)
        plt.savefig(self.path, format=self.format, dpi=80, bbox_inches='tight')
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def plot_origin(self):
        data = self.data
        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()
        # bplot = ax.boxplot(x=data.values, labels=data.columns, whis=1.5, medianprops=dict(color='k'))
        bplot = ax.boxplot(x=data.values,
                           labels=data.columns,
                           patch_artist=True,  # fill with color
                           medianprops=dict(color='k', linewidth=2))

        # Fill with colors: Tableau Palette
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        colors = ['lightskyblue', 'khaki', 'lightgreen', 'salmon', 'mediumpurple', 'sandybrown']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(labels=data.columns, fontsize=14)
        ax.set_ylabel("CFV", fontsize=16)
        # ax.set_title('Box', fontsize=25)
        plt.savefig(self.path, format=self.format, dpi=80)
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
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(labels=data.columns, fontsize=10)
        ax.set_ylabel("CFV Error", fontsize=16)
        # ax.set_title('Box', fontsize=25)
        plt.savefig(self.path, format=self.format, dpi=80)
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
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(labels=data.columns, fontsize=10)
        ax.set_ylabel("Log(CFV Error)", fontsize=16)
        # ax.set_title('Box', fontsize=25)
        plt.savefig(self.path, format=self.format, dpi=80)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    from benchmarks import Sphere
    csv_path = r"../../tests/mf_param_opt_tests/parameter_evaluation_tests/output/MultiRunsTest/MFOptimizerCS/DE.csv"
    box_path = r"output/PlotBox/DE.png"
    multi_runs_data = pd.read_csv(csv_path, header=0, index_col=0)
    multi_runs_data.drop(['mean', 'std', 'median', 'best', 'worst'], inplace=True)
    multi_runs_data.drop(['Raw'], axis=1, inplace=True)

    pb = PlotBox(data=multi_runs_data, benchmark=Sphere(dimension=50), path=box_path, show=True)
    # pb.plot_seaborn()
    pb.plot_edge()
    # pb.plot_origin()
    # pb.plot_error()
    # pb.plot_log()
