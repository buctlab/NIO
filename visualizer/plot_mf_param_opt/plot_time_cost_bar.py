import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange, array
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
        data = array([0, 0, 0])
        data[1:] = self.data['Time Cost'].values

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        width = 0.5

        xticks = self.data.index
        n = data.shape[0]
        ind = arange(n)
        data = data / 3600
        colors = ['black', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        plt.bar(x=ind, height=data, width=width, color=colors)

        ax.set_xticks(ind[1:])
        ax.set_xticklabels(xticks)

        # ax.set_xlabel('Multi-fidelity control strategy', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_ylabel('Time Cost (h)', fontsize=16)

        if self.show_flag:
            plt.show()
        fig.savefig(self.path, format=self.format, dpi=80, bbox_inches='tight')
