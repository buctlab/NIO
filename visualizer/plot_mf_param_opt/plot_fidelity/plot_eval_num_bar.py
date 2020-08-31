import matplotlib.pyplot as plt
from numpy import arange
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotEvalNumBar')
logger.setLevel('INFO')


class PlotEvalNumBar:

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
        data = self.data

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        width = 0.35

        xticks = data['Fidelity']
        n = data.shape[0]
        ind = arange(n)  # the x locations for the groups
        plt.bar(x=ind, height=data['EvalCounts'], width=width, color='orange', alpha=0.5, edgecolor='black', tick_label=xticks)

        ax.set_xticks(ind)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Fidelity Level', fontsize=16)
        ax.set_ylabel('Eval Num', fontsize=16)
        # ax.set_title("Evaluation Num of 10-Level Fidelity", fontsize=16)

        plt.savefig(self.path, format=self.format, dpi=80)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    csv_path = r"../../../tests/mf_param_opt_tests/output/FidelityLevelTest.csv"
    bar_path = r"output/PlotEvalNumBar/FidelityLevelEvalNum.png"
    data = pd.read_csv(csv_path, header=0, index_col=0)

    pen = PlotEvalNumBar(data=data[['Fidelity', 'EvalCounts']], path=bar_path, show=True)
    pen.plot()
