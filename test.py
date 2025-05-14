from simulated_bifurcation.models import ABCModel
from typing import Union, Optional
import numpy as np
import torch
from monaco.mc_sim import Sim
from sklearn import covariance
import cvxpy
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from model import helper

data = {
    'Stock #1': [12,13,15,16,18,12,14,15,12,14,18,19],
    'Stock #2': [20,22,21,20,22,23,24,21,20,22,21,20],
    'Stock #3': [10,12,13,12,12,11,10,15,12,14,16,12],
    'Stock #4': [15,16,15,14,15,17,18,17,17,16,15,14],
    'Stock #5': [22,22,21,22,23,24,25,23,24,25,26,27],
    'Stock #6': [18,15,16,17,19,21,24,22,23,21,20,19]
}
prices = {
    'Stock #1': [16],
    'Stock #2': [22],
    'Stock #3': [12],
    'Stock #4': [16],
    'Stock #5': [23],
    'Stock #6': [19],
}
theta = 100
R = 75
h = helper(theta,R,data,prices,K=3)
h.helper_optimize()