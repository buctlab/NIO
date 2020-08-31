import benchmarks
import matplotlib
matplotlib.use('Agg')
import unittest
import multiprocessing
import inspect
import logging
import os

logging.basicConfig()
logger = logging.getLogger('Plot3DBenchmarkTest')
logger.setLevel('INFO')


class Plot3DBenchmarkTest(unittest.TestCase):

    def setUp(self):
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        self.cpu = multiprocessing.cpu_count()  # 8
        self.benchmarks = []
        self.path = r"output/Plot3DBenchmarkTest"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.tasks = list()

        for benchname, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and benchname not in ['Benchmark']:
                self.benchmarks.append(benchmark())

    def tearDown(self):
        logger.info(self.cpu)
        for task in self.tasks:
            task.get()
        logger.info("Plot 3D-Benchmarks Done!")

    def test_plot_all_benchmarks(self):
        for benchmark in self.benchmarks:
            task = self.pool.apply_async(func=benchmark.plot, args=(abs(benchmark.upper[0]/20), self.path))
            self.tasks.append(task)
        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    unittest.main()
