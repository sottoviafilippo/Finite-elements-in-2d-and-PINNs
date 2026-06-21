# inspiration (and theory) from arXiv:2403.00599v1

import numpy as np
from numpy.random import uniform
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
from scipy.stats import qmc


class PINN_Poisson_2d:
    """Simpler case: solve the Poisson equation in 2d with given Dirichlet boundary conditions"""

    def __init__(self, N_internal_nodes: int, f_dirichlet: Callable):
        # first version: 3 internal layers, hardcoded for the sake of simplicity
        self.N_internal_nodes  = N_internal_nodes

        # Why Tanh? for instance ReLU would not work since its second derivatives vanish (and the Poisson equation is built on them)
        self.model = nn.Sequential(
            nn.Linear(2, 32),  
            nn.Tanh(),         
            nn.Linear(32, 32),  
            nn.Tanh(),
            nn.Linear(32, 32),  
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        pass

    def set_collocation_points(self, N_collocation_points: int, x_bounds: tuple, y_bounds: tuple):
        """Sets the collocation points to be later used in the optimization procedure"""

        sampler = qmc.LatinHypercube(d = 2) # 2d model, 2 dimensions
        standard_samples = sampler.random(n = N_collocation_points)

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        lower_bounds = [x_min, y_min]
        upper_bounds = [x_max, y_max]

        self.collocation_points = qmc.scale(standard_samples, lower_bounds, upper_bounds)

        pass 



class PINN_heat_2d:

    def __init__(self):
        pass



    