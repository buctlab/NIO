import unittest
import multiprocessing
import benchmarks
import algorithms
import pandas as pd
import inspect
import logging
import os

logging.basicConfig()
logger = logging.getLogger('ConvergenceTest')
logger.setLevel('INFO')


class ConvergenceTest(unittest.TestCase):

    def setUp(self):
        self.iterations = 200
        self.path = r"output/AllTests2019-12/convergence"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.benchmarks = list()
        self.benchnames = list()
        self.algorithms = list()
        self.algonames = list()
        self.results = None

        for name, algorithm in inspect.getmembers(algorithms):
            if inspect.isclass(algorithm) and name not in ['Algorithm', 'KrillHerdBase', 'GeneticAlgorithm',
                                                           'MyFakeAlgorithm', 'RandomCalculation']:
                # if inspect.isclass(algorithm) and name in ['CuckooSearch']:
                self.algorithms.append(algorithm)
                self.algonames.append(name)
        print(self.algonames)

    def save_res(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for bench_index, row in self.results.iterrows():
            convergence = pd.DataFrame(columns=self.algonames, index=list(range(1, self.iterations + 1)))
            for alg_index in self.algonames:
                try:
                    convergence[alg_index] = row[alg_index].get()
                except ValueError as e:
                    logger.error(alg_index + ": " + str(e))
            # logger.info(convergence)
            csv_path = "{path}/{bench}.csv".format(path=path, bench=bench_index)
            convergence.to_csv(csv_path)
            logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_2d_benchmarks(self):
        dim = 2
        self.benchmarks = list()
        self.benchnames = list()
        for name, benchmark in inspect.getmembers(benchmarks):
            # if inspect.isclass(benchmark) and name != "Benchmark":
            if inspect.isclass(benchmark) and name == "Ackley":
                self.benchmarks.append(benchmark)
                self.benchnames.append(name)

        self.results = pd.DataFrame(columns=self.algonames, index=self.benchnames, dtype=object)
        multiprocessing.freeze_support()
        self.pool = multiprocessing.Pool()
        for i in range(len(self.benchmarks)):
            benchmark = self.benchmarks[i](dimension=dim)
            for j in range(len(self.algorithms)):
                alg = self.algorithms[j](func=benchmark, iterations=self.iterations)
                self.results.iloc[i, j] = self.pool.apply_async(func=alg.run_return_convergence)
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D".format(dim=dim)
        self.save_res(path)

    def test_5d_michalewicz(self):
        dim = 5
        benchmark = benchmarks.Michalewicz(dimension=dim)
        convergence = pd.DataFrame(columns=self.algonames, index=list(range(1, self.iterations + 1)))
        for i in range(len(self.algorithms)):
            alg = self.algorithms[i](func=benchmark, iterations=self.iterations)
            convergence[self.algonames[i]] = alg.run_return_convergence()

        path = self.path + r"/{dim}D".format(dim=dim)
        if not os.path.exists(path):
            os.makedirs(path)
        csv_path = "{path}/{bench}.csv".format(path=path, bench=benchmark.__class__.__name__)
        convergence.to_csv(csv_path)
        logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_10d_michalewicz(self):
        dim = 10
        benchmark = benchmarks.Michalewicz(dimension=dim)
        convergence = pd.DataFrame(columns=self.algonames, index=list(range(1, self.iterations + 1)))
        for i in range(len(self.algorithms)):
            alg = self.algorithms[i](func=benchmark, iterations=self.iterations)
            convergence[self.algonames[i]] = alg.run_return_convergence()

        path = self.path + r"/{dim}D".format(dim=dim)
        if not os.path.exists(path):
            os.makedirs(path)
        csv_path = "{path}/{bench}.csv".format(path=path, bench=benchmark.__class__.__name__)
        convergence.to_csv(csv_path)
        logger.info("Success Generate {test}: {file}".format(test=self._testMethodName, file=csv_path))

    def test_30d_benchmarks(self):
        dim = 30
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        # high_dim_bench = ['Ackley']
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
                self.results.iloc[i, j] = self.pool.apply_async(func=alg.run_return_convergence)
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D_cs".format(dim=dim)
        self.save_res(path)

    def test_50d_benchmarks(self):
        self.iterations = 200
        dim = 50
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        # high_dim_bench = ['Stybtang']
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
                self.results.iloc[i, j] = self.pool.apply_async(func=alg.run_return_convergence)
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D_ad".format(dim=dim)
        self.save_res(path)

    def test_100d_benchmarks(self):
        self.iterations = 200
        dim = 100
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        # high_dim_bench = ['Schwefel', 'Sphere', 'Stybtang']
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
                self.results.iloc[i, j] = self.pool.apply_async(func=alg.run_return_convergence)
        self.pool.close()
        self.pool.join()

        path = self.path + r"/{dim}D_ad".format(dim=dim)
        self.save_res(path)


if __name__ == '__main__':
    unittest.main()
