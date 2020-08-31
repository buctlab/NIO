import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBar')
logger.setLevel('INFO')


class PlotBar:

    def __init__(self, data, path, show=True):
        self.data = data
        self.path = path
        self.show_flag = show
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def autolabel(ax, rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')

    def plot_bar(self):
        data = self.data

        # Create new Figure and an Axes which fills it.
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        N = len(data[0])
        ind = arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        rects1 = ax.bar(ind - width/2, data[0], width, color='SkyBlue', label='Time Cost')
        rects2 = ax.bar(ind + width/2, data[1], width, color='IndianRed', label='Eval Counts')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title("Param Opt: Non-MultiFidelity VS MultiFidelity")
        ax.set_xticks(ind)
        # ax.set_xticklabels(('CS', 'DE', 'KH', 'SSA', 'WWO'))
        ax.set_xticklabels(('Non-MultiFidelity', 'MultiFidelity'))
        ax.legend()

        self.autolabel(ax, rects1, "center")
        self.autolabel(ax, rects2, "center")

        if self.show_flag:
            plt.show()
        fig.savefig(self.path, dpi=200)


if __name__ == '__main__':
    csv_path = r"../tests/.csv"
    bar_path = r"output/PlotBar/ParamOptCompare.png"
    # data = pd.read_csv(csv_path, header=0, index_col=0)
    data = [[58908900, 96905400], [10872360, 18216360]]

    pb = PlotBar(data=data, path=bar_path)
    pb.plot_bar()
