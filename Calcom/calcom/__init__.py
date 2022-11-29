# -*- coding: utf-8 -*-

"""Top-level package for CalCom."""

__author__ = """Manuchehr Aminian, Tarequl Islam Sifat"""
__email__ = 'aminian@colostate.edu'
__version__ = '0.4.0'


# from .calcom import Calcom
from .experiment import Experiment
from .ccexperiment import CCExperiment
from .parameter_search.grid_search import GridSearch
from .parameter_search.bayesian_optimization import BayesianOptimization
from . import solvers, plot_wrapper

from . import datasets

# top-level convenience functions
from .calcom import load_style
