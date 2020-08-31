from algorithms.algorithm import Algorithm, Ackley
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger('ALO')
logger.setLevel('INFO')


class RandomCalculation(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitness = None
        self.best = None

    def run(self):
        result = np.random.uniform(self.func.lower, self.func.upper, self.func.dimension)
        self.fitness = self.cost_function(result)
        self.best = self.fitness
        count = 0
        while not self.stopping_criteria_precision(self.eval_count, self.fitness):
            result = np.random.uniform(self.func.lower, self.func.upper, self.func.dimension)
            self.fitness = self.cost_function(result)
            # print(self.eval_count, self.stopping_eval)
            if self.fitness < self.best:
                self.best = self.fitness
            # print("eval:{eval}, sol:{sol},fit:{fitness}".format(eval=count, sol=result,fitness=self.fitness))
            count += 1

    def run_return_best_val(self):
        result = np.random.uniform(self.func.lower, self.func.upper,self.func.dimension)
        self.fitness = self.cost_function(result)
        self.best = self.fitness
        count = 0
        while not self.stopping_criteria_eval():
            result = np.random.uniform(self.func.lower, self.func.upper, self.func.dimension)
            print(round(result[0], 5))
            self.fitness = self.cost_function(result)
            if self.fitness < self.best:
                self.best = self.fitness
            count+=1


if __name__ == '__main__':
    ran = RandomCalculation()
    ran.run_return_best_val()
    print(ran.eval_count, "{best}".format(best=ran.best))

