import benchmarks
from visualizer import PlotBox
import unittest
import pandas as pd
import multiprocessing
import inspect
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotBoxTest')
logger.setLevel('INFO')


AlgMap = {
    'AntLionOptimizer': 'ALO',
    'BatAlgorithm': 'BA',
    'CuckooSearch': 'CS',
    'DifferentialEvolution': 'DE',
    'DispersiveFliesOptimisation': 'DFO',
    'FireflyAlgorithm': 'FA',
    'FlowerPollinationAlgorithm': 'FPA',
    'FruitFly': 'FOA',
    'GeneticAlgorithm': 'GA',
    'GreyWolfOptimizer': 'GWO',
    'KrillHerd': 'KH',
    'MothFlameOptimization': 'MFO',
    'ParticleSwarmOptimization': 'PSO',
    'SalpSwarmAlgorithm': 'SSA\nSalp',
    'SquirrelSearchAlgorithm': 'SSA\nSquirrel',
    'WaterWaveOptimization': 'WWO',
    'WhaleOptimizationAlgorithm': 'WOA',
}


class PlotBoxTest(unittest.TestCase):

    def setUp(self):
        self.data = list()
        self.box_path = list()
        self.path = r"output/PlotBoxTest"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.benchmarks = list()
        self.benchnames = list()

        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name != 'Benchmarks':
                self.benchnames.append(name)
                self.benchmarks.append(benchmark)
                self.box_path.append(r"{path}/{bench}.png".format(path=self.path, bench=name))

        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        self.cpu = multiprocessing.cpu_count()

        self.plot_origin, self.plot_log, self.plot_error = list(), list(), list()

    def tearDown(self):
        logger.info(self.cpu)

        for i in range(len(self.benchmarks)):
            self.plot_origin[i].get()
            self.plot_log[i].get()
            self.plot_error[i].get()
            logger.info("Success Generate Box: {bench}.png {bench}_log.png {bench}_error.png".format(
                bench=self.benchnames[i]))
        logger.info("Plot Boxes Done!")

    def data_processing(self):
        for bench in self.benchnames:
            csv_path = r"../algorithm_parallel_tests/output/MultiRunsTest/{bench}.csv".format(bench=bench)
            data = pd.read_csv(csv_path, header=0, index_col=0)
            data.rename(columns=AlgMap, inplace=True)
            data.drop(['mean', 'std'], inplace=True)

            self.data.append(data)

    def test_plot_all_box(self):
        self.data_processing()

        for i in range(len(self.benchmarks)):
            pb = PlotBox(data=self.data[i], benchmark=self.benchmarks[i]().get_optimum()[-1], path=self.box_path[i], show=False)

            self.plot_origin.append(self.pool.apply_async(func=pb.plot_origin, args=()))
            self.plot_log.append(self.pool.apply_async(func=pb.plot_log, args=()))
            self.plot_error.append(self.pool.apply_async(func=pb.plot_error, args=()))

        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    unittest.main()
