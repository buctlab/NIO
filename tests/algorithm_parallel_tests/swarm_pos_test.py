import unittest
import multiprocessing
import benchmarks
import algorithms
import pandas as pd
import inspect
import logging
import os

logging.basicConfig()
logger = logging.getLogger('SwarmPosTest')
logger.setLevel('INFO')


class SwarmPosTest(unittest.TestCase):

    def setUp(self):
        self.iterations = 200
        self.path = r"output/AllTests2019-12/SwarmPosTest"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.benchmarks = list()
        self.benchnames = list()
        self.algorithms = list()
        self.algonames = list()
        self.results = None

        for name, algorithm in inspect.getmembers(algorithms):
             if inspect.isclass(algorithm) and name not in ['Algorithm', 'KrillHerdBase', 'GeneticAlgorithm']:
            # if inspect.isclass(algorithm) and name in ['SailfishOptimizer']:
                self.algorithms.append(algorithm)
                self.algonames.append(name)

    def save_res(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for bench_index, row in self.results.iterrows():
            for alg_index in self.algonames:
                swarm_pos = None
                try:
                    swarm_pos = row[alg_index].get()
                except ValueError as e:
                    logger.error(alg_index + ": " + str(e))
                swarm_pos = pd.DataFrame(swarm_pos)
                # logger.info("\n%s %s:\n%s" % (bench_index, algo_index, swarm_pos))
                csv_path = "{path}/{alg}_{bench}.csv".format(path=path, alg=alg_index, bench=bench_index)
                swarm_pos.to_csv(csv_path)
                logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_2d_benchmarks(self):
        dim = 2
        self.benchmarks = list()
        self.benchnames = list()
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name not in [ 'Benchmark']:
            # if inspect.isclass(benchmark) and name == 'Michalewicz':
                self.benchmarks.append(benchmark)
                self.benchnames.append(name)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                alg = self.algorithms[j](func=benchmark, iterations=self.iterations)
                self.results.iloc[i, j] = self.pool.apply_async(func=alg.run_return_swarm_pos)
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D".format(dim=dim)
        self.save_res(path)

    def test_5d_michalewicz(self):
        dim = 5
        path = self.path + r"/{dim}D".format(dim=dim)
        if not os.path.exists(path):
            os.makedirs(path)

        benchmark = benchmarks.Michalewicz(dimension=dim)
        for i in range(len(self.algorithms)):
            alg = self.algorithms[i](func=benchmark, iterations=self.iterations)
            swarm_pos = alg.run_return_swarm_pos()
            csv_path = "{path}/{alg}_{bench}.csv".format(path=path, alg=self.algonames[i], bench=benchmark.__class__.__name__)
            swarm_pos.to_csv(csv_path)
            logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_10d_michalewicz(self):
        dim = 10
        path = self.path + r"/{dim}D".format(dim=dim)
        if not os.path.exists(path):
            os.makedirs(path)

        benchmark = benchmarks.Michalewicz(dimension=dim)
        for i in range(len(self.algorithms)):
            alg = self.algorithms[i](func=benchmark, iterations=self.iterations)
            swarm_pos = alg.run_return_swarm_pos()
            csv_path = "{path}/{alg}_{bench}.csv".format(path=path, alg=self.algonames[i],
                                                         bench=benchmark.__class__.__name__)
            swarm_pos.to_csv(csv_path)
            logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_30d_benchmarks(self):
        dim = 30
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        self.benchmarks = list()
        self.benchnames = list()
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name in high_dim_bench:
                self.benchmarks.append(benchmark)
                self.benchnames.append(name)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                alg = self.algorithms[j](func=benchmark, iterations=self.iterations)
                self.results.iloc[i, j] = self.pool.apply_async(func=alg.run_return_swarm_pos)
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D".format(dim=dim)
        self.save_res(path)


if __name__ == '__main__':
    unittest.main()
