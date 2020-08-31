from matplotlib import pyplot as plt
import mpl_toolkits.axisartist as axisartist
from numpy import linspace, vectorize
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotFidelityFunction')
logger.setLevel('INFO')


def Linear(x):
    if x >= 1:
        return 10
    return x*10


class PlotFidelityFunction:

    def __init__(self, func, path, show=False):
        self.func = func

        self.show_flag = show
        self.path = path
        (filepath, tempfilename) = os.path.split(path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        (filename, extension) = os.path.splitext(tempfilename)
        self.format = extension[1:]

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        # ax.set_xlabel("Gen/nGen", fontsize=25)
        # ax.set_ylabel("Fidelity", fontsize=25)
        ax.axis["bottom"].label.set(text="x", size=20)
        ax.axis["left"].label.set(text="f(x)", size=20)
        ax.axis["bottom"].major_ticklabels.set_size(16)
        ax.axis["left"].major_ticklabels.set_size(16)

        # ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["bottom"].set_axisline_style("->", size=1.5)
        ax.axis["left"].set_axisline_style("->", size=1.5)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)
        x = linspace(0, 1, 11)
        y = vectorize(self.func)(x)
        plt.plot(x, y, marker='o', color='k')
        plt.savefig(self.path, format=self.format, dpi=80)
        if self.show_flag:
            plt.show()


if __name__ == '__main__':
    path = r"output/PlotFidelityFunction/{fig}.eps".format(fig=Linear.__name__)
    pff = PlotFidelityFunction(Linear, path, show=True)
    pff.plot()
