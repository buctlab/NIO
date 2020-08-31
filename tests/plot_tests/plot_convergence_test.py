import benchmarks
from visualizer import PlotConvergence
import unittest
import pandas as pd
import multiprocessing
import inspect
import os
import logging

logging.basicConfig()
logger = logging.getLogger('PlotConvergenceTest')
logger.setLevel('INFO')


AlgMap = {
    'AntLion': 'ALO',
    'Bat': 'BA',
    'CuckooSearch': 'CS',
    'DifferentialEvolution': 'DE',
    'DispersiveFlies': 'DFO',
    'Firefly': 'FA',
    'FlowerPollination': 'FPA',
    'FruitFly': 'FOA',
    'GeneticAlgorithm': 'GA',
    'GreyWolfOptimization': 'GWO',
    'KrillHerd': 'KH',
    'MothFlame': 'MFO',
    'ParticalSwarm': 'PSO',
    'SalpSwarm': 'SSA\nSalp',
    'SquirrelSearchAlgorithm': 'SSA\nSquirrel',
    'WaterWaveOptimization': 'WWO',
    'WhaleOptimization': 'WOA',
}


class PlotConvergenceTest(unittest.TestCase):

    def setUp(self):
        self.data = list()
        self.convergence_figure_path = list()
        self.path = r"output/PlotConvergenceTest"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.benchmarks = list()
        self.benchnames = list()

        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name != 'Benchmarks':
                self.benchnames.append(name)
                self.benchmarks.append(benchmark)
                self.convergence_figure_path.append(r"{path}/{bench}.png".format(path=self.path, bench=name))

        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        self.cpu = multiprocessing.cpu_count()

        self.plot_convergence = list()

    def tearDown(self):
        logger.info(self.cpu)

        for i in range(len(self.benchmarks)):
            self.plot_convergence[i].get()
            logger.info("Success Generate Convergence Figure: {bench}.png".format(bench=self.benchnames[i]))
        logger.info("Plot Convergence Figures Done!")

    def data_processing(self):
        for bench in self.benchnames:
            csv_path = r"../algorithm_parallel_tests/output/ConvergenceTest/{bench}.csv".format(bench=bench)
            data = pd.read_csv(csv_path, header=0, index_col=0)
            # data.rename(columns=AlgMap, inplace=True)
            self.data.append(data)

    def test_plot_all_convergence(self):
        self.data_processing()

        for i in range(len(self.benchmarks)):
            pc = PlotConvergence(data=self.data[i], baseline_val=self.benchmarks[i]().get_optimum()[-1],
                                 benchname=self.benchmarks[i]().__class__.__name__,
                                 path=self.convergence_figure_path[i], show=False)
            self.plot_convergence.append(self.pool.apply_async(func=pc.plot_convergence, args=()))

        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    unittest.main()
