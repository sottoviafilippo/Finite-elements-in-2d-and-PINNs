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

    def __init__(self, N_internal_nodes: int, f_poisson: Callable, f_dirichlet: Callable, x_bounds: tuple, y_bounds: tuple):
        """
        Poisson equation: d_xx u + d_yy u = -f_poisson. Dirichlet b.c.: u(bdy) = f_dirichlet
        """
        # first version: 3 internal layers, hardcoded for the sake of simplicity
        self.N_internal_nodes = N_internal_nodes
        self.f_dirichlet = f_dirichlet
        self.f_poisson = f_poisson

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        # Why Tanh? for instance ReLU would not work since its second derivatives vanish (and the Poisson equation is built on them)
        self.model = nn.Sequential(
            nn.Linear(2, N_internal_nodes),  
            nn.Tanh(),         
            nn.Linear(N_internal_nodes, N_internal_nodes),  
            nn.Tanh(),
            nn.Linear(N_internal_nodes, N_internal_nodes),  
            nn.Tanh(),
            nn.Linear(N_internal_nodes, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        pass

    def set_collocation_points(self, N_collocation_points: int):
        """Sets the collocation points to be later used in the optimization procedure"""

        sampler = qmc.LatinHypercube(d = 2) # 2d model, 2 dimensions
        standard_samples = sampler.random(n = N_collocation_points)

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        lower_bounds = [x_min, y_min]
        upper_bounds = [x_max, y_max]

        self.collocation_points = torch.tensor(qmc.scale(standard_samples, lower_bounds, upper_bounds), dtype=torch.float32, requires_grad=True)

        pass 

    def compute_boundary_values(self, N_boundary_points: int):
        """Computes the boundary data used later in the optimization process, in order to impose the Dirichlet b.c. 
        Same number of points on every segment
        """

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
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

    def __init__(self, N_internal_nodes: int, f_initial: Callable, f_dirichlet: Callable, x_bounds: tuple, y_bounds: tuple, t_bounds: tuple, alpha: float = 1):
        """
        Heat equation: alpha(d_xx u + d_yy u) = d_t u. Dirichlet b.c.: u(boundary) = f(boundary) at all times. Initial b.c.: u(t=0) = f_initial
        """
        # first version: 4 internal layers, hardcoded for the sake of simplicity
        # Larger network than for the 2d Poisson equation.
        self.N_internal_nodes = N_internal_nodes
        self.f_dirichlet = f_dirichlet
        self.f_initial = f_initial
        self.alpha = alpha

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.t_bounds = t_bounds
         
        # input order: x, y, t
        self.model = nn.Sequential(
            nn.Linear(3, N_internal_nodes),  
            nn.Tanh(),         
            nn.Linear(N_internal_nodes, N_internal_nodes),  
            nn.Tanh(),
            nn.Linear(N_internal_nodes, N_internal_nodes),
            nn.Tanh(),
            nn.Linear(N_internal_nodes, N_internal_nodes),  
            nn.Tanh(),
            nn.Linear(N_internal_nodes, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        pass


    def set_collocation_points(self, N_collocation_points: int):
        """Sets the collocation points to be later used in the optimization procedure"""

        sampler = qmc.LatinHypercube(d = 3) # 2+1d model, 3 dimensions
        standard_samples = sampler.random(n = N_collocation_points)

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        t_min, t_max = self.t_bounds
        lower_bounds = [x_min, y_min, t_min]
        upper_bounds = [x_max, y_max, t_max]

        self.collocation_points = torch.tensor(qmc.scale(standard_samples, lower_bounds, upper_bounds), dtype=torch.float32, requires_grad=True)

        pass 


    def compute_boundary_values(self, N_boundary_points: int):
        """Computes the boundary data used later in the optimization process, in order to impose the Dirichlet b.c. 
        Same number of points on every segment
        """

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        t_min, t_max = self.t_bounds
        
        # number of points on each side proportional to the side lengths. May pose some problems if one side is much shorter
        N_points_horizontal = int(np.abs((x_max - x_min)/(x_max - x_min + y_max - y_min))*N_boundary_points/2)
        N_points_vertical   = int(N_boundary_points/2) - N_points_horizontal
 
        self.boundary_points = [[random.uniform(x_min, x_max), y_min, random.uniform(t_min, t_max)] for k in range(N_points_horizontal)] + [[random.uniform(x_min, x_max), y_max, random.uniform(t_min, t_max)] for k in range(N_points_horizontal)] + [[x_min, random.uniform(y_min, y_max), random.uniform(t_min, t_max)] for k in range(N_points_vertical)] + [[x_max, random.uniform(y_min, y_max), random.uniform(t_min, t_max)] for k in range(N_points_vertical)]
        self.boundary_values = torch.tensor([self.f_dirichlet(p) for p in self.boundary_points], dtype=torch.float32).reshape(-1, 1)
        self.boundary_points = torch.tensor(self.boundary_points, dtype=torch.float32) # don't need requires_grad = True here, since these are Dirichlet b.c.

        pass


    def compute_initial_values(self, N_initial_points: int, t_min: float):
        """Initial conditions at t=t_min (typically t=0)
        """

        sampler = qmc.LatinHypercube(d=2)  # 2+1d model, 2 spatial dimensions
        standard_samples = sampler.random(n=N_initial_points)

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        lower_bounds = [x_min, y_min]
        upper_bounds = [x_max, y_max]

        scaled_samples = qmc.scale(standard_samples, lower_bounds, upper_bounds)

        t_col = np.full((N_initial_points, 1), t_min) # add t_min because these are the initial conditions
        scaled_samples_with_t = np.hstack([scaled_samples, t_col])
       
        self.initial_values = torch.tensor([self.f_initial(p) for p in scaled_samples_with_t], dtype=torch.float32).reshape(-1, 1)
        self.initial_points = torch.tensor(scaled_samples_with_t, dtype=torch.float32, requires_grad=True)


    def compute_physics_loss(self):
        """Computes the physics loss based on the heat equation alpha(d_xx u + d_yy u) - d_t u = 0"""

        u = self.model(self.collocation_points)

        # first compute the first derivatives
        grad_u = torch.autograd.grad(outputs=u, inputs=self.collocation_points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        # now compute the second derivatives
        u_xx = torch.autograd.grad(outputs=u_x, inputs=self.collocation_points, grad_outputs=torch.ones_like(u_x),create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(outputs=u_y, inputs=self.collocation_points, grad_outputs=torch.ones_like(u_y),create_graph=True, retain_graph=True)[0][:, 1:2]

        return torch.mean((self.alpha * (u_xx + u_yy) - u_t) ** 2) # beware of sign


    def compute_dirichlet_loss(self):

        criterion = nn.MSELoss()
        predictions = self.model(self.boundary_points)

        return criterion(predictions, self.boundary_values)
    

    def compute_initial_loss(self):
        """loss corresponding to the initial conditions at t = t_min"""

        criterion = nn.MSELoss()
        predictions = self.model(self.initial_points)

        return criterion(predictions, self.initial_values)


    def train(self, N_epochs, weight_bdy = 10., weight_in = 10.):

        self.physics_losses   = []
        self.dirichlet_losses = []
        self.initial_losees   = []
        self.epochs           = []

        for epoch in range(N_epochs):
            self.epochs.append(epoch + 1)
            phys_loss = self.compute_physics_loss()
            bdy_loss  = self.compute_dirichlet_loss()
            in_loss   = self.compute_initial_loss()
            loss      = phys_loss + weight_bdy * bdy_loss + weight_in * in_loss

            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step() 

            self.physics_losses.append(phys_loss.item())
            self.dirichlet_losses.append(bdy_loss.item())

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{N_epochs}], Loss: {loss.item():.4f}")


    def _compute_residuals_at_points(self, points: torch.Tensor):
        # Computes the pointwise squared PDE residual at the given points.
        # points: leaf tensor with requires_grad=True.
        # Used by train_RARG to rank candidate points by how badly they violate the PDE.
        
        u = self.model(points)
        grad_u = torch.autograd.grad(outputs=u, inputs=points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=points, grad_outputs=torch.ones_like(u_x),create_graph=False, retain_graph=True)[0][:, 0:1]
        # only retain_graph on u_x since the graph feeding is shared (they both come from grad_u)
        u_yy = torch.autograd.grad(outputs=u_y, inputs=points, grad_outputs=torch.ones_like(u_y),create_graph=False, retain_graph=False)[0][:, 1:2]
        residual = (self.alpha * (u_xx + u_yy) - u_t) ** 2  # shape (N, 1), per-point squared residual

        return residual.detach()

    
    def train_RARG(self, N_new_points = 500, m = 50, max_epochs = 30000, max_points = 20000, N_epochs_every_collocation_set = 250,  weight_bdy = 10., weight_in = 10.):
        # residual-based adaptive finement with greed. Following https://arxiv.org/pdf/2207.10289
        # m: number of points to be added at every iteration (out of N_new_points sampled)
        # N_epochs_every_collocation_set: number of iterations to be performed with every number of collocation points

        self.physics_losses   = []
        self.dirichlet_losses = []
        self.initial_losees   = []
        N_epochs              = 0

        sampler = qmc.LatinHypercube(d = 3) # 2+1d model, 3 dimensions

        # first train on the LHS-sample collocation points
        for epoch in range(N_epochs_every_collocation_set):
            N_epochs +=1
            phys_loss = self.compute_physics_loss()
            bdy_loss  = self.compute_dirichlet_loss()
            in_loss   = self.compute_initial_loss()
            loss      = phys_loss + weight_bdy * bdy_loss + weight_in * in_loss

            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step() 

            self.physics_losses.append(phys_loss.item())
            self.dirichlet_losses.append(bdy_loss.item())

        while N_epochs <= max_epochs and len(self.collocation_points) <= max_points:
            # first, sample N_new_points new collocation points
            standard_samples = sampler.random(n = N_new_points)
            x_min, x_max = self.x_bounds
            y_min, y_max = self.y_bounds
            t_min, t_max = self.t_bounds
            lower_bounds = [x_min, y_min, t_min]
            upper_bounds = [x_max, y_max, t_max]
            
            candidate_points = torch.tensor(qmc.scale(standard_samples, lower_bounds, upper_bounds),dtype=torch.float32, requires_grad=True)

            # evaluate the PDE residual at each candidate point
            residuals_candidate_points = self._compute_residuals_at_points(candidate_points).squeeze(-1)  # shape (N_new_points,)

            # only keep the m new collocation points with the largest residuals ("greedy" selection)
            m_eff = min(m, N_new_points)
            _, topk_indices = torch.topk(residuals_candidate_points, m_eff)
            new_points = candidate_points[topk_indices].detach()

            # merge with the existing collocation points, making sure grads are tracked again
            self.collocation_points = torch.cat([self.collocation_points.detach(), new_points], dim=0).requires_grad_(True)

            # now train

            for epoch in range(N_epochs_every_collocation_set):
                N_epochs += 1
                phys_loss = self.compute_physics_loss()
                bdy_loss  = self.compute_dirichlet_loss()
                in_loss   = self.compute_initial_loss()
                loss      = phys_loss + weight_bdy * bdy_loss + weight_in * in_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.physics_losses.append(phys_loss.item())
                self.dirichlet_losses.append(bdy_loss.item())
                self.initial_losees.append(in_loss.item())

                if N_epochs % 1000 == 0:
                    print(f"Epoch [{N_epochs}/{max_epochs}], N_collocation_points: {len(self.collocation_points)}, Loss: {loss.item():.4f}")

                if N_epochs > max_epochs:
                    break


# TO DO: optimal dimensions (number of layers and nodes) for both cases
# what is the optimal number of collocation and boundary points?
# what weight should be given to the boundary loss?  