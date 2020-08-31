import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy import zeros_like, meshgrid, arange
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBar3D')
logger.setLevel('INFO')


class PlotBar3D:

    def __init__(self, data, path, show=True):
        self.data = data
        self.path = path
        self.show_flag = show
        (path, filename) = os.path.split(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_bar3d(self):
        # setup the figure and axes
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        data = self.data

        _x = data.index.values
        _y = data.columns.values
        _xx, _yy = meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = -data.values.ravel()
        bottom = zeros_like(top)
        width = depth = 1

        ax.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax.set_title('Mean of result at 10-levels fidelity')

        if self.show_flag:
            plt.show()
        fig.savefig(self.path, dpi=200)


if __name__ == '__main__':
    ssa_csv_path = r"../tests/parameter_evaluation_tests/output/MFSSAParamEvaluationTest.csv"
    bar3d_path = r"output/PlotBar3D/Mean10LevelFidelity.png"
    ssa_data = pd.read_csv(ssa_csv_path, index_col=0)
    ssa_mean = ssa_data.loc['mean'].rename('SSA')

    data = pd.DataFrame(index=['SSA1', 'SSA2'], columns=list(range(1, 11)))
    data.loc['SSA1'] = ssa_data.loc['mean'].values
    data.loc['SSA2'] = ssa_data.loc['mean'].values

    pb3d = PlotBar3D(data=data, path=bar3d_path)
    pb3d.plot_bar3d()
