import unittest
from benchmarks import *
import algorithms
import numpy as np
import pandas as pd
import inspect
import logging
import os
from time import time

logging.basicConfig()
logger = logging.getLogger('RunTimeTest')
logger.setLevel('INFO')


class RunTimeTest(unittest.TestCase):

    def setUp(self):
        self.iterations = 500
        self.dimension = 5
        self.benchmark = Michalewicz(lower=[0] * self.dimension, upper=[np.pi] * self.dimension,
                                     dimension=self.dimension)
        self.path = r"output/RunTimeTest"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.algorithms = []
        self.algonames = []

        for name, algorithm in inspect.getmembers(algorithms):
            if inspect.isclass(algorithm) and name != 'Algorithm':
                self.algorithms.append(algorithm)
                self.algonames.append(name)

        self.times = pd.Series(index=self.algonames)

    def tearDown(self):
        logger.info(self.times.sort_values())
        csv_path = r"{path}/RunTimeRank.csv".format(path=self.path)
        self.times.sort_values().to_csv(csv_path)

    def test_algorithm_runtime(self):
        tasks = list()

        for i in range(len(self.algorithms)):
            task = self.algorithms[i](func=self.benchmark, iterations=self.iterations)
            tasks.append(task)
        for i in range(len(tasks)):
            begin = time()
            tasks[i].run()
            end = time()
            self.times[self.algonames[i]] = end - begin


if __name__ == '__main__':
    unittest.main()
