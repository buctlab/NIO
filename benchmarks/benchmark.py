import abc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from numpy import inf, arange, meshgrid, vectorize, full, zeros, array, ndarray
from matplotlib import cm


class Benchmark(metaclass=abc.ABCMeta):

    def __init__(self, lower, upper, dimension):
        self.dimension = dimension
        if isinstance(lower, (float, int)):
            self.lower = full(self.dimension, lower)
            self.upper = full(self.dimension, upper)
        elif isinstance(lower, (ndarray, list, tuple)) and len(lower) == dimension:
            self.lower = array(lower)
            self.upper = array(upper)
        else:
            raise ValueError("{bench}: Type mismatch or Length of bound mismatch with dimension".format(bench=self.__class__.__name__))

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0
        pass

    @staticmethod
    def eval(**kwargs):
        return inf
        pass

    def __2dfun(self, x, y, f): return f((x, y))

    def plot(self, scale=None, save_path=None):
        if not scale:
            scale = abs(self.upper[0] / 100)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        func = self.eval
        X_range, Y_range = arange(self.lower[0], self.upper[0], scale), arange(self.lower[1], self.upper[1], scale)
        X, Y = meshgrid(X_range, Y_range)
        Z = vectorize(self.__2dfun)(X, Y, func)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6, cmap=cm.rainbow)
        # cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if save_path:
            plt.savefig(save_path+'/{benchmark}.png'.format(benchmark=self.__class__.__name__), dpi=100)
            plt.clf()
            plt.close()
        else:
            plt.show()
        # plt.show()
