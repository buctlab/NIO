from algorithms.algorithm import Algorithm
from algorithms.ant_lion_optimizer import AntLionOptimizer
from algorithms.bat_algorithm import BatAlgorithm
from algorithms.cuckoo_search import CuckooSearch
from algorithms.differential_evolution import DifferentialEvolution
from algorithms.dispersive_flies_optimisation import DispersiveFliesOptimisation
from algorithms.firefly_algorithm import FireflyAlgorithm
from algorithms.flower_pollination_algorithm import FlowerPollinationAlgorithm
from algorithms.fruitfly_optimization_algorithm import FruitFly
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.grey_wolf_optimizer import GreyWolfOptimizer
from algorithms.krill_herd import KrillHerdBase, KrillHerd
from algorithms.moth_flame_optimization import MothFlameOptimization
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.salp_swarm_algorithm import SalpSwarmAlgorithm
from algorithms.squirrel_search_algorithm import SquirrelSearchAlgorithm
from algorithms.water_wave_optimization import WaterWaveOptimization
from algorithms.whale_optimization_algorithm import WhaleOptimizationAlgorithm
# from algorithms.random_calculation import RandomCalculation
from algorithms.rooted_tree_optimization import RootedTreeOptimization
from algorithms.black_widow_optimization_algorithm import BlackWidowOptimizationAlgorithm
from algorithms.sailfish_optimizer import SailfishOptimizer

__all__ = [
    'Abbreviation',
    'Algorithm',
    'AntLionOptimizer',
    'BatAlgorithm',
    'CuckooSearch',
    'DifferentialEvolution',
    'DispersiveFliesOptimisation',
    'FireflyAlgorithm',
    'FlowerPollinationAlgorithm',
    'FruitFly',
    'GeneticAlgorithm',
    'GreyWolfOptimizer',
    'KrillHerd',
    'MothFlameOptimization',
    'ParticleSwarmOptimization',
    'SalpSwarmAlgorithm',
    'SquirrelSearchAlgorithm',
    'WaterWaveOptimization',
    'WhaleOptimizationAlgorithm',
    # 'RandomCalculation',
    'RootedTreeOptimization',
    'BlackWidowOptimizationAlgorithm',
    'SailfishOptimizer'
]

Abbreviation = {
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
    'SalpSwarmAlgorithm': 'S(a)SA',
    'SquirrelSearchAlgorithm': 'S(q)SA',
    'WaterWaveOptimization': 'WWO',
    'WhaleOptimizationAlgorithm': 'WOA',
    "RootedTreeOptimization": 'RTO',
    "BlackWidowOptimizationAlgorithm": "BWOA",
    "SailfishOptimizer": "SFO"
}
