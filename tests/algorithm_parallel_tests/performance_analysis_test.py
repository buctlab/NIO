import unittest
import logging
import os
import benchmarks
from algorithms import Abbreviation as Abbr
import inspect
import pandas as pd
import numpy as np

logging.basicConfig()
logger = logging.getLogger('ConvergenceTest')
logger.setLevel('INFO')


class PerformanceAnalysis(unittest.TestCase):

    def setUp(self):
        self.path = r"output/HighDim/performance"
        self.res_path = r"output/HighDim/performance/results"
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

        self.benchmarks = list()
        self.bench_names = list()

    @staticmethod
    def data_process(data):
        data.rename(columns=Abbr, inplace=True)
        return data

    def test_2D_benchmarks(self):
        dim = 2
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name != 'Benchmark':
                # if inspect.isclass(benchmark) and name == 'Ackley':
                self.benchmarks.append(benchmark)
                self.bench_names.append(name)

        csv_path = self.path + r"/{dim}D".format(dim=dim)

        df = pd.DataFrame()
        res_file = self.res_path + r"/{dim}D.csv".format(dim=dim)

        for i in range(len(self.benchmarks)):
            bench_name = self.bench_names[i]
            csv_file = r"{path}/{bench}.csv".format(path=csv_path, bench=bench_name)
            data = pd.read_csv(csv_file, header=0, index_col=0)
            res = data.loc[:, ['min_algo', 'min_value']]
            if res.loc['min_iter', 'min_value'] < np.iinfo(np.int32).max - 100:
                res.loc['error', 'min_algo'] = ""
                res.loc['error', 'min_value'] = ""

            res.index = pd.MultiIndex.from_product([[self.bench_names[i]], ['min_iter', 'error']])
            df = df.append(res)
            df.to_csv(res_file)
        logger.info(df)

    def test_5D_benchmarks(self):
        dim = 10

        csv_path = self.path + r"/{dim}D".format(dim=dim)

        df = pd.DataFrame()
        res_file = self.res_path + r"/{dim}D.csv".format(dim=dim)

        bench_name = 'Michalewicz'
        csv_file = r"{path}/{bench}.csv".format(path=csv_path, bench=bench_name)
        data = pd.read_csv(csv_file, header=0, index_col=0)
        res = data.loc[:, ['min_algo', 'min_value']]
        if res.loc['min_iter', 'min_value'] < np.iinfo(np.int32).max - 100:
            res.loc['error', 'min_algo'] = ""
            res.loc['error', 'min_value'] = ""
        else:
            res.loc['min_iter', 'min_algo'] = ""
            res.loc['min_iter', 'min_value'] = ""

        res.index = pd.MultiIndex.from_product([[bench_name + "-{dim}D".format(dim=dim)], ['min_iter', 'error']])
        df = df.append(res)
        df.to_csv(res_file)

    def test_30D_benchmarks(self):
        dim = 100
        high_dim_bench = ['Ackley', 'Griewank', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Sphere', 'Stybtang']
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name in high_dim_bench:
                self.benchmarks.append(benchmark)
                self.bench_names.append(name)

        csv_path = self.path + r"/{dim}D".format(dim=dim)

        df = pd.DataFrame()
        res_file = self.res_path + r"/{dim}D.csv".format(dim=dim)

        for i in range(len(self.benchmarks)):
            bench_name = self.bench_names[i]
            csv_file = r"{path}/{bench}.csv".format(path=csv_path, bench=bench_name)
            data = pd.read_csv(csv_file, header=0, index_col=0)
            res = data.loc[:, ['min_algo', 'min_value']]
            if res.loc['min_iter', 'min_value'] < np.iinfo(np.int32).max - 100:
                res.loc['error', 'min_algo'] = ""
                res.loc['error', 'min_value'] = ""
            else:
                res.loc['min_iter', 'min_algo'] = ""
                res.loc['min_iter', 'min_value'] = ""

            res.index = pd.MultiIndex.from_product(
                [[self.bench_names[i] + "-{dim}D".format(dim=dim)], ['min_iter', 'error']])
            df = df.append(res)
            df.to_csv(res_file)
        logger.info(df)

    def test_sum_2D(self):
        csv_path = self.path + r"/results"
        dims = [2, 5, 10, 30, 50, 100]
        # dims = [2]
        for i in range(len(dims)):
            csv_file = r"{path}/{dim}D.csv".format(path=csv_path, dim=dims[i])
            df = pd.read_csv(csv_file)
            print("\n")
            print("-----------{dim}---------------".format(dim=dims[i]))
            # print(df.loc[:, 'min_algo'].value_counts())
            print(df)

    def test_new_df_2D(self):
        for name, benchmark in inspect.getmembers(benchmarks):
            if inspect.isclass(benchmark) and name != 'Benchmark':
                self.benchmarks.append(benchmark)
                self.bench_names.append(name)

        df = pd.DataFrame(data=np.zeros((len(self.bench_names) * 2, 2)),
                          index=pd.MultiIndex.from_product([self.bench_names, ['min_iter', 'error']]),
                          columns=['min_algo', 'min_value'])
        if df.loc[('Ackley', 'min_iter'), 'min_value'] == 0:
            df.loc[('Ackley', 'min_iter'), 'min_value'] = 1

        print(df)
