import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotTimeCost')
logger.setLevel('INFO')


class PlotTimeCostBar:

    def __init__(self, data, path, show=False):
        self.data = data
        self.path = path
        self.show_flag = show
        (filepath, tempfilename) = os.path.split(path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        (filename, extension) = os.path.splitext(tempfilename)
        self.format = extension[1:]

    def plot(self):
        data = self.data  # pd.DataFrame

        fig = plt.figure(figsize=(10, 6))
        # Create new Figure and an Axes which fills it.
        ax = fig.add_subplot(111)
        width = 0.35  # the width of the bars: can also be len(x) sequence

        xticks = data['Fidelity']
        n = data.shape[0]
        ind = arange(n)  # the x locations for the groups
        # data['TimeCost'] = data['TimeCost'].apply(lambda x: x / 3600)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        # colors = ['lightgrey', 'lightgray', 'silver', 'darkgrey', 'darkgray', 'grey', 'gray', 'dimgrey', 'dimgray', 'black']
        # plt.bar(x=ind, height=data['TimeCost'], width=width, color=colors, alpha=0.5, edgecolor='black', tick_label=xticks)
        plt.bar(x=ind, height=data['TimeCost'], width=width, color=colors, tick_label=xticks)
        # for i, v in enumerate(data):
        #     ax.text(ind, v+2, str(v))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Fidelity Level', fontsize=16)
        ax.set_ylabel('Time Cost (s)', fontsize=16)
        # ax.legend()
        # ax.set_title("Time Cost of 10-Level Fidelity", fontsize=16)

        if self.show_flag:
            plt.show()
        fig.savefig(self.path, format=self.format, dpi=80)

    def plot_plume(self):
        data = self.data  # pd.DataFrame

        fig = plt.figure(figsize=(10, 6))
        # Create new Figure and an Axes which fills it.
        ax = fig.add_subplot(111)
        width = 0.35  # the width of the bars: can also be len(x) sequence

        # xticks = data['Fidelity']
        xticks = data.index
        n = data.shape[0]
        ind = arange(n)  # the x locations for the groups
        # data['TimeCost'] = data['TimeCost'].apply(lambda x: x / 3600)
        # plt.bar(x=ind, height=data['Fidelity'], width=width, color='orange', alpha=0.5, edgecolor='black', tick_label=xticks)
        data = data / 3600
        colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        # colors = ['khaki', 'lightgreen', 'salmon', 'mediumpurple', 'sandybrown']
        # colors = ['darkgray', 'lightgreen', 'salmon', 'mediumpurple', 'sandybrown']
        plt.bar(x=ind, height=data, width=width, color=colors, tick_label=xticks)
        # for i, v in enumerate(data):
        #     ax.text(ind, v+2, str(v))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks)
        # ax.set_xlabel('Fidelity Level', fontsize=16)
        ax.set_xlabel('Multi-fidelity control strategy', fontsize=16)
        ax.set_ylabel('Time Cost (h)', fontsize=16)
        # ax.legend()
        # ax.set_title("Time Cost of 10-Level Fidelity", fontsize=16)

        if self.show_flag:
            plt.show()
        fig.savefig(self.path, format=self.format, dpi=80)


if __name__ == '__main__':
    csv_path = r"../../../tests/mf_param_opt_tests/output/FidelityLevelTest.csv"
    bar_path = r"output/PlotTimeCostBar/FidelityLevelTimeCost.png"
    data = pd.read_csv(csv_path, header=0, index_col=0)

    ptc = PlotTimeCostBar(data=data[['Fidelity', 'TimeCost']], path=bar_path, show=True)
    ptc.plot()
