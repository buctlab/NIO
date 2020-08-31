import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBar')
logger.setLevel('INFO')


class PlotBar:

    def __init__(self, data, title, path, show=True):
        self.data = data
        self.title = title
        self.path = path
        self.show_flag = show
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_bar(self):
        data = self.data
        xticks = data.index

        plt.figure(figsize=(4, 3))
        # Create new Figure and an Axes which fills it.
        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars: can also be len(x) sequence

        n = data.shape[0]
        ind = arange(n)
        if isinstance(data, pd.Series):
            n = data.shape[0]
            ind = arange(n)  # the x locations for the groups
            plt.bar(x=ind, height=data, width=width, tick_label=xticks)
        elif isinstance(data, pd.DataFrame):
            n = data.shape[1]
            ind = arange(n)  # the x locations for the groups
            xticks = data.columns
            plt.bar(x=ind, height=data.loc['mean'], width=width, tick_label=xticks, yerr=data.loc['std'])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(ind)
        # ax.set_xticklabels(xticks)
        ax.set_xlabel('Fidelity')
        ax.set_ylabel('Value')
        # ax.legend()
        ax.set_title(self.title)

        if self.show_flag:
            plt.show()
        fig.savefig(self.path, dpi=160)


if __name__ == '__main__':
    csv_path = r"../tests/parameter_evaluation_tests/output/MFPSOParamEvaluationTest.csv"
    bar_median_path = r"output/PlotBar/FidelityEvalMedianBar.png"
    bar_mean_path = r"output/PlotBar/FidelityEvalMeanBar.png"
    data = pd.read_csv(csv_path, header=0, index_col=0)

    pb = PlotBar(data=data.loc['median'], path=bar_median_path, title="10-FidelityLevel run 100 Iterations Median Result")
    pb.plot_bar()

    pb = PlotBar(data=data.loc[['mean', 'std']], path=bar_mean_path, title="10-FidelityLevel run 100 Iterations Mean & Std Result")
    pb.plot_bar()
