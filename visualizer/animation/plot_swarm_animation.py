import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import linspace, meshgrid, asarray, inf
from scipy.spatial.distance import euclidean as ed
import pandas as pd
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotSwarmAnimation')
logger.setLevel('INFO')


class PlotSwarmAnimation:

    def __init__(self, data, benchmark, path, show=False):
        self.frames = data.index.levels[0].size

        self.data = data
        if benchmark is not None:
            self.benchmark = benchmark
            self.lower, self.upper = self.benchmark.lower, self.benchmark.upper
            self.ref_points = asarray(self.benchmark.get_optimum()[0])
        else:
            self.benchmark = None
        self.threshold = 10e-2

        self.path = path
        self.show_flag = show

        (path, filename) = os.path.split(path)
        if self.path is not None and not os.path.exists(path):
            os.makedirs(path)

        self.animation = None

    def plot(self):
        # Create new Figure and an Axes which fills it.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.subplots()
        ax.set_title('Evolutionary Progress')

        # Plot contour of benchmark
        if self.benchmark is not None:
            n = 256
            x = linspace(self.lower[0], self.upper[0], n)
            y = linspace(self.lower[1], self.upper[1], n)
            X, Y = meshgrid(x, y)
            Z = self.benchmark.eval([X, Y])
            ax.contour(X, Y, Z, 8, alpha=0.75, cmap=plt.get_cmap('gray'))

        iter_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        # Construct the scatter which we will update during animation as the swarm pos for each iteration
        data = self.data
        # iterations = len(data.index.get_level_values('Iteration').unique())
        # frames = linspace(0, iterations, self.frames).tolist()
        scat = ax.scatter(data.loc[1].iloc[:, 0], data.loc[1].iloc[:, 1], c='g', s=20, marker='o', label="Individual Position")

        if self.benchmark is not None:
            # data.loc[1]: The boundary point causes the side to be white. Set the fixed coordinate display range.
            plt.axis([self.lower[0], self.upper[0], self.lower[1], self.upper[1]])
            # logger.info(plt.axis())

        # Construct the animation, using the update function as the animation director.
        self.animation = FuncAnimation(fig, self.update_scatter, frames=self.frames, fargs=(data, scat, iter_text), interval=50, repeat=False)

        # Reference best pos marked with star
        if self.benchmark is not None and self.ref_points.shape[1] > 0:
            ax.scatter(self.ref_points[:, 0], self.ref_points[:, 1], c='r', s=100, marker='*', label="Global Minimum")
        ax.legend(loc='upper right')
        if self.show_flag:
            plt.show()

    def color_calculation(self, one_pos):
        dis_min = inf
        for ref_point in self.ref_points:
            dis = ed(ref_point, one_pos)
            dis_min = dis if dis < dis_min else dis_min
        return 'g' if dis_min < self.threshold else 'g'

    def update_scatter(self, frame_number, data, scat, iter_text):
        # index of current frame(iteration)
        index = frame_number+1

        # new position for current iteration
        swarm_pos = data.loc[index]
        if self.benchmark is not None:
            colors = swarm_pos.apply(self.color_calculation, axis=1)
            # print(colors)
            scat.set_facecolors(colors)

        # Update the scatter collection, with the new colors, sizes and positions.
        # scat.set_edgecolors(colors)
        # scat.set_sizes(sizes)
        # scat.set_facecolors(colors)
        scat.set_offsets(swarm_pos)
        iter_text.set_text('Gen: {i}'.format(i=index))

        return scat, iter_text

    def save(self):
        self.animation.save(self.path, dpi=200)


if __name__ == '__main__':
    from benchmarks import Ackley, Michalewicz
    csv_path = r"../../tests/algorithm_parallel_tests/output/SwarmPosTest/2D/WaterWaveOptimization_Michalewicz.csv"
    swarm_animation_path = r"output/PlotSwarmAnimation/WWO_Michalewicz.mp4"
    swarm_pos_data = pd.read_csv(csv_path, index_col=[0, 1], header=[0])

    pas = PlotSwarmAnimation(data=swarm_pos_data, benchmark=Michalewicz(dimension=2), path=swarm_animation_path)
    pas.plot()
    pas.save()
