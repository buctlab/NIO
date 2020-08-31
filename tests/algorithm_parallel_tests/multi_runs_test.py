import unittest
import multiprocessing
import benchmarks
import algorithms
from numpy import random, zeros, mean, std, median, min, max
import pandas as pd
import inspect
import logging
import os

logging.basicConfig()
logger = logging.getLogger('MultipleTest')
logger.setLevel('INFO')


class MultiRunsTest(unittest.TestCase):

    def setUp(self):
        self.runs = 30
        self.iterations = 200
        self.Rand = random.RandomState(seed=1)
        self.path = r"output/HighDim/multipleTests"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.benchmarks = list()
        self.benchnames = list()
        self.algorithms = list()
        self.algonames = list()
        self.results = None
        self.eval = None

        for name, algorithm in inspect.getmembers(algorithms):
            # if inspect.isclass(algorithm) and name not in ['Algorithm', 'KrillHerdBase', 'GeneticAlgorithm' ]:
            if inspect.isclass(algorithm) and name in ['CuckooSearch']:
                self.algorithms.append(algorithm)
                self.algonames.append(name)

    def save_res(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for bench_index, row in self.results.iterrows():
            multi_runs = pd.DataFrame(columns=self.algonames, index=list(range(1, self.runs + 1)))
            for alg_index in self.algonames:
                res = row[alg_index]  # array length = self.runs
                for r in range(self.runs):
                    try:
                        multi_runs.loc[r + 1, alg_index] = res[r].get()
                    except ValueError as e:
                        logger.error(alg_index + ": " + str(e))
            multi_runs.loc['mean'] = multi_runs.apply(mean)
            multi_runs.loc['std'] = multi_runs.apply(std)
            multi_runs.loc['median'] = multi_runs.iloc[:self.runs, :].apply(median)
            multi_runs.loc['best'] = multi_runs.iloc[:self.runs, :].apply(min)
            multi_runs.loc['worst'] = multi_runs.iloc[:self.runs, :].apply(max)
            # logger.info("\n{}".format(multi_runs))
            csv_path = "{path}/{bench}.csv".format(path=path, bench=bench_index)
            multi_runs.to_csv(csv_path)
            logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_2d_benchmarks(self):
        dim = 2
        self.benchmarks = list()
        self.benchnames = list()
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name != 'Benchmark':
                # if inspect.isclass(benchmark) and name == 'Michalewicz':
                self.benchmarks.append(benchmark)
                self.benchnames.append(name)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                res = zeros(shape=self.runs, dtype=object)
                for r in range(self.runs):
                    alg = self.algorithms[j](func=benchmark, seed=self.Rand.randint(0, 10 * self.runs),
                                             iterations=self.iterations)
                    res[r] = self.pool.apply_async(func=alg.run_return_best_val)

                self.results.iloc[i, j] = res  # array length = self.runs
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D".format(dim=dim)
        self.save_res(path)

    def test_5d_michalewicz(self):
        dim = 5
        self.benchmarks = [benchmarks.Michalewicz]
        self.benchnames = ['Michalewicz']
        print(self.algonames)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                res = zeros(shape=self.runs, dtype=object)
                for r in range(self.runs):
                    alg = self.algorithms[j](func=benchmark, seed=self.Rand.randint(0, 10 * self.runs),
                                             iterations=self.iterations)
                    res[r] = self.pool.apply_async(func=alg.run_return_best_val)
                self.results.iloc[i, j] = res  # array length = self.runs
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D".format(dim=dim)
        self.save_res(path)

    def test_10d_michalewicz(self):
        dim = 10
        self.benchmarks = [benchmarks.Michalewicz]
        self.benchnames = ['Michalewicz']

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                res = zeros(shape=self.runs, dtype=object)
                for r in range(self.runs):
                    alg = self.algorithms[j](func=benchmark, seed=self.Rand.randint(0, 10 * self.runs),
                                             iterations=self.iterations)
                    res[r] = self.pool.apply_async(func=alg.run_return_best_val)
                self.results.iloc[i, j] = res  # array length = self.runs
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D".format(dim=dim)
        self.save_res(path)

    def test_100d_benchmarks(self):
        dim = 100
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        # high_dim_bench = ['Ackley']
        self.benchmarks = list()
        self.benchnames = list()
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name in high_dim_bench:
                self.benchmarks.append(benchmark)
                self.benchnames.append(name)
        print(self.algonames)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                res = zeros(shape=self.runs, dtype=object)
                for r in range(self.runs):
                    alg = self.algorithms[j](func=benchmark, seed=self.Rand.randint(0, 10 * self.runs),
                                             iterations=self.iterations)
                    res[r] = self.pool.apply_async(func=alg.run_return_best_val)
                self.results.iloc[i, j] = res
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D_CS".format(dim=dim)
        self.save_res(path)

    def test_50d_benchmarks(self):
        dim = 50
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        # high_dim_bench = ['Ackley']
        self.benchmarks = list()
        self.benchnames = list()
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name in high_dim_bench:
                self.benchmarks.append(benchmark)
                self.benchnames.append(name)
        print(self.algonames)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                res = zeros(shape=self.runs, dtype=object)
                for r in range(self.runs):
                    alg = self.algorithms[j](func=benchmark, seed=self.Rand.randint(0, 10 * self.runs),
                                             iterations=self.iterations)
                    res[r] = self.pool.apply_async(func=alg.run_return_best_val)
                self.results.iloc[i, j] = res
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D_CS".format(dim=dim)
        self.save_res(path)


if __name__ == '__main__':
    unittest.main()
