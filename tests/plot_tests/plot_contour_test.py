import algorithms
import benchmarks
from visualizer import PlotSwarmPos2D
import unittest
import pandas as pd
import multiprocessing
import inspect
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotContourTest')
logger.setLevel('INFO')


class PlotContourTest(unittest.TestCase):

    def setUp(self):
        self.iterations = [0, 100, 200]
        self.path = r"output/PlotContourTest"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.algorithms, self.algonames = list(), list()
        self.benchmarks, self.benchnames = list(), list()

        for name, algorithm in inspect.getmembers(algorithms):
            if inspect.isclass(algorithm) and name != 'Algorithm':
                self.algorithms.append(algorithm)
                self.algonames.append(name)
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name != 'Benchmarks':
                self.benchnames.append(name)
                self.benchmarks.append(benchmark)

        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        self.cpu = multiprocessing.cpu_count()

        self.plot_contour = list()

    def tearDown(self):
        logger.info(self.cpu)

        for i in range(len(self.plot_contour)):
            self.plot_contour[i].get()
        logger.info("Plot Contours Done!")

    def test_plot_all_contour(self):

        for i in self.iterations:
            csv_path = r"../algorithm_parallel_tests/output/SwarmPosTest/Iteration{iter}".format(iter=i)
            contour_path = r"{path}/Iteration{iter}".format(path=self.path, iter=i)
            for j in range(len(self.algorithms)):
                for k in range(len(self.benchmarks)):
                    csv_file = r"{path}/{alg}_{bench}.csv".format(path=csv_path,
                                                                  alg=self.algonames[j], bench=self.benchnames[k])
                    contour_file = r"{path}/{alg}_{bench}.csv".format(path=contour_path,
                                                                      alg=self.algonames[j], bench=self.benchnames[k])
                    data = pd.read_csv(csv_file, header=0, index_col=0)

                    pc = PlotSwarmPos2D(data=data, benchmark=self.benchmarks[k], path=contour_file, show=False)
                    self.plot_contour.append(self.pool.apply_async(func=pc.plot_contour, args=()))

        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    unittest.main()
