# inspiration (and theory) from arXiv:2403.00599v1

import numpy as np
from numpy.random import uniform
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
from scipy.stats import qmc
import random


class PINN_Poisson_2d:
    """Simpler case: solve the Poisson equation in 2d with given Dirichlet boundary conditions"""

    def __init__(self, N_internal_nodes: int, f_poisson: Callable, f_dirichlet: Callable):
        """
        Poisson equation: d_xx u + d_yy u = f_poisson. Dirichlet b.c.: u(bdy) = f_dirichlet
        """
        # first version: 3 internal layers, hardcoded for the sake of simplicity
        self.N_internal_nodes  = N_internal_nodes
        self.f_dirichlet = f_dirichlet

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

    def compute_boundary_values(self, N_boundary_points: int, x_bounds: tuple, y_bounds: tuple):
        """Computes the boundary data used later in the optimization process, in order to impose the Dirichlet b.c. 
        Same number of points on every segment
        """

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        N_points_horizontal = int(np.abs((x_max - x_min)/(x_max - x_min + y_max - y_min))*N_boundary_points/2)
        N_points_vertical   = int(N_boundary_points/2) - N_points_horizontal
 
        self.boundary_points = [[random.uniform(x_min, x_max), y_min] for k in range(N_points_horizontal)] + [[random.uniform(x_min, x_max), y_max] for k in range(N_points_horizontal)] + [[x_min, random.uniform(y_min, y_max)] for k in range(N_points_vertical)] + [[x_max, random.uniform(y_min, y_max)] for k in range(N_points_vertical)]
        self.boundary_values = [self.f_dirichlet(p) for p in self.boundary_points]
        pass

    def compute_physics_loss(self):
        """Computes the physics loss based on the Poisson equation d_xx u + d_yy u = f"""

class PINN_heat_2d:

    def __init__(self):
        pass



    