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
        Poisson equation: d_xx u + d_yy u = -f_poisson. Dirichlet b.c.: u(bdy) = f_dirichlet
        """
        # first version: 3 internal layers, hardcoded for the sake of simplicity
        self.N_internal_nodes = N_internal_nodes
        self.f_dirichlet = f_dirichlet
        self.f_poisson = f_poisson

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

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        pass

    def set_collocation_points(self, N_collocation_points: int, x_bounds: tuple, y_bounds: tuple):
        """Sets the collocation points to be later used in the optimization procedure"""

        sampler = qmc.LatinHypercube(d = 2) # 2d model, 2 dimensions
        standard_samples = sampler.random(n = N_collocation_points)

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        lower_bounds = [x_min, y_min]
        upper_bounds = [x_max, y_max]

        self.collocation_points = torch.tensor(qmc.scale(standard_samples, lower_bounds, upper_bounds), dtype=torch.float32, requires_grad=True)

        pass 

    def compute_boundary_values(self, N_boundary_points: int, x_bounds: tuple, y_bounds: tuple):
        """Computes the boundary data used later in the optimization process, in order to impose the Dirichlet b.c. 
        Same number of points on every segment
        """

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        # maybe one should also add cornerns for strict boundary enforcement?
        N_points_horizontal = int(np.abs((x_max - x_min)/(x_max - x_min + y_max - y_min))*N_boundary_points/2)
        N_points_vertical   = int(N_boundary_points/2) - N_points_horizontal
 
        self.boundary_points = [[random.uniform(x_min, x_max), y_min] for k in range(N_points_horizontal)] + [[random.uniform(x_min, x_max), y_max] for k in range(N_points_horizontal)] + [[x_min, random.uniform(y_min, y_max)] for k in range(N_points_vertical)] + [[x_max, random.uniform(y_min, y_max)] for k in range(N_points_vertical)]
        self.boundary_values = torch.tensor([self.f_dirichlet(p) for p in self.boundary_points], dtype=torch.float32).reshape(-1, 1)
        self.boundary_points = torch.tensor(self.boundary_points, dtype=torch.float32) # don't need requires_grad = True here, since these are Dirichlet b.c.

        pass

    def compute_physics_loss(self):
        """Computes the physics loss based on the Poisson equation d_xx u + d_yy u = f_poisson"""

        u = self.model(self.collocation_points)

        # first compute the first derivatives
        grad_u = torch.autograd.grad(outputs=u, inputs=self.collocation_points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]

        # now compute the second derivatives
        u_xx = torch.autograd.grad(outputs=u_x, inputs=self.collocation_points, grad_outputs=torch.ones_like(u_x),create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(outputs=u_y, inputs=self.collocation_points, grad_outputs=torch.ones_like(u_y),create_graph=True, retain_graph=True)[0][:, 1:2]

        f = self.f_poisson(self.collocation_points)  # note that f_Poisson should be defined with torch functions

        return torch.mean((u_xx + u_yy + f) ** 2) # laplacian = -f (beware of sign)
    
    def compute_dirichlet_loss(self):

        criterion = nn.MSELoss()
        predictions = self.model(self.boundary_points)

        return criterion(predictions, self.boundary_values)


    def train(self, N_epochs, weight_bdy = 10.):

        self.physics_losses = []
        self.dirichlet_losses = []
        self.epochs = []

        for epoch in range(N_epochs):
            self.epochs.append(epoch + 1)
            phys_loss = self.compute_physics_loss()
            bdy_loss = self.compute_dirichlet_loss()
            loss = phys_loss + weight_bdy * bdy_loss

            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step() 

            self.physics_losses.append(phys_loss.item())
            self.dirichlet_losses.append(bdy_loss.item())

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{N_epochs}], Loss: {loss.item():.4f}")




class PINN_heat_2d:
    """Solve the heat equation in 2 spatial dimensions + time"""

    def __init__(self, N_internal_nodes: int, f_initial: Callable, f_boundary: Callable, alpha: float = 1):
        """
        Heat equation: alpha(d_xx u + d_yy u) = d-t u. Dirichlet b.c.: u(boundary) = f(boundary) at all times. Initial b.c.: u(t=0) = f_initial
        """
        # first version: 4 internal layers, hardcoded for the sake of simplicity
        # Larger network than for the 2d Poisson equation.
        self.N_internal_nodes = N_internal_nodes
        self.f_boundary = f_boundary
        self.f_initial = f_initial
         
        self.model = nn.Sequential(
            nn.Linear(3, 48),  
            nn.Tanh(),         
            nn.Linear(48, 48),  
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 48),  
            nn.Tanh(),
            nn.Linear(48, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        pass




# TO DO: optimal dimensions (number of layers and nodes) for both cases
# what is the optimal number of collocation and boundary points?
# what weight should be given to the boundary loss?  