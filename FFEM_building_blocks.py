import numpy as np
import warnings

class Mesh:
    """
    Produces a 2D finite element computational grid and computes associated 
    matrices for the resolution of differential equations.
    
    Uses a triangular mesh with linear elements.
    """

    def __init__(self, x_positions: np.ndarray, y_positions: np.ndarray):
        """Initializes the class with position arrays and creates the grid."""
        self.x_pos = x_positions
        self.y_pos = y_positions
        self.grid = np.meshgrid(x_positions, y_positions)

        # Compute the number of grid points in both directions
        self.Nx = len(self.x_pos)
        self.Ny = len(self.y_pos)

    def find_neighbors(self, x_index: int, y_index: int):
        """Finds the list of neighbors of a given grid point."""
        if x_index < 0 or y_index < 0:
            warnings.warn("Warning: negative index", UserWarning)
            return 0
        
        if x_index >= self.Nx or y_index >= self.Ny:
            warnings.warn("Warning: index out of range", UserWarning)
            return 0
        
        all_possible_neighbors = [
            [x_index - 1, y_index], 
            [x_index, y_index + 1], 
            [x_index + 1, y_index], 
            [x_index, y_index - 1]
        ]

        # Only return elements within the grid boundaries
        return [
            nb for nb in all_possible_neighbors 
            if 0 <= nb[0] < self.Nx and 0 <= nb[1] < self.Ny
        ]

    def check_in_grid(self, x_index: int, y_index: int):
        """ Checks whether a given set of indices is in the grid"""
        return 0 <= x_index < self.Nx and 0 <= y_index < self.Ny

    def compute_integral_of_basis_function(self, x_index: int, y_index: int):
        """Computes the diagonal elements of \int \psi_i \psi_j."""
        if x_index < 0 or y_index < 0:
            warnings.warn("Warning: negative index", UserWarning)
            return 0
        
        if x_index >= self.Nx or y_index >= self.Ny:
            warnings.warn("Warning: index out of range", UserWarning)
            return 0

        neighbors = self.find_neighbors(x_index, y_index)
        n_nb = len(neighbors)
        inte = 0.0  # Initialize integral value

        if n_nb == 4:
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 6.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 12.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 6.0
            inte += (self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 12.0

        elif x_index == 0:
            if y_index == 0:
                inte = (self.x_pos[1] - self.x_pos[0]) * (self.y_pos[1] - self.y_pos[0]) / 12.0
            elif y_index == self.Ny - 1:
                inte = (self.x_pos[1] - self.x_pos[0]) * (self.y_pos[-1] - self.y_pos[-2]) / 6.0
            else:
                inte = (self.x_pos[1] - self.x_pos[0]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 12.0
                inte += (self.x_pos[1] - self.x_pos[0]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 6.0
        
        elif x_index == self.Nx - 1:
            if y_index == self.Ny - 1:
                inte = (self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[-1] - self.y_pos[-2]) / 12.0
            elif y_index == 0:
                inte = (self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[1] - self.y_pos[0]) / 6.0
            else:
                inte = (self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 12.0
                inte += (self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 6.0

        elif y_index == 0:
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[1] - self.y_pos[0]) / 6.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[1] - self.y_pos[0]) / 12.0

        elif y_index == self.Ny - 1:
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[-1] - self.y_pos[-2]) / 12.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[-1] - self.y_pos[-2]) / 6.0

        return inte
    
    def compute_integral_of_basis_function_dx(self, x_index: int, y_index: int):
        """Computes the diagonal elements of \int d_x\psi_i d_x\psi_j."""
        if x_index < 0 or y_index < 0:
            warnings.warn("Warning: negative index", UserWarning)
            return 0
        
        if x_index >= self.Nx or y_index >= self.Ny:
            warnings.warn("Warning: index out of range", UserWarning)
            return 0

        neighbors = self.find_neighbors(x_index, y_index)
        n_nb = len(neighbors)
        inte = 0.0  # Initialize integral value

        if n_nb == 4:
            inte += 1/(self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
            inte += 1/(self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
            inte += 1/(self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
            inte += 1/(self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0

        elif x_index == 0:
            if y_index == 0:
                inte = 1/(self.x_pos[1] - self.x_pos[0]) * (self.y_pos[1] - self.y_pos[0]) / 2.0
            elif y_index == self.Ny - 1:
                inte = 1/(self.x_pos[1] - self.x_pos[0]) * (self.y_pos[-1] - self.y_pos[-2]) / 2.0
            else:
                inte = 1/(self.x_pos[1] - self.x_pos[0]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
                inte += 1/(self.x_pos[1] - self.x_pos[0]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
        
        elif x_index == self.Nx - 1:
            if y_index == self.Ny - 1:
                inte = 1/(self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[-1] - self.y_pos[-2]) / 2.0
            elif y_index == 0:
                inte = 1/(self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[1] - self.y_pos[0]) / 2.0
            else:
                inte = 1/(self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
                inte += 1/(self.x_pos[-1] - self.x_pos[-2]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0

        elif y_index == 0:
            inte = 1/(self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[1] - self.y_pos[0]) / 2.0
            inte += 1/(self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[1] - self.y_pos[0]) / 2.0

        elif y_index == self.Ny - 1:
            inte = 1/(self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[-1] - self.y_pos[-2]) / 2.0
            inte += 1/(self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[-1] - self.y_pos[-2]) / 2.0

        return inte

    def compute_integral_of_basis_function_dy(self, x_index: int, y_index: int):
        """Computes the diagonal elements of \int d_y\psi_i d_y\psi_j."""
        if x_index < 0 or y_index < 0:
            warnings.warn("Warning: negative index", UserWarning)
            return 0
        
        if x_index >= self.Nx or y_index >= self.Ny:
            warnings.warn("Warning: index out of range", UserWarning)
            return 0

        neighbors = self.find_neighbors(x_index, y_index)
        n_nb = len(neighbors)
        inte = 0.0  # Initialize integral value

        if n_nb == 4:
            inte += (self.x_pos[x_index] - self.x_pos[x_index - 1]) / (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) / (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) / (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
            inte += (self.x_pos[x_index] - self.x_pos[x_index - 1]) / (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0

        elif x_index == 0:
            if y_index == 0:
                inte = (self.x_pos[1] - self.x_pos[0]) / (self.y_pos[1] - self.y_pos[0]) / 2.0
            elif y_index == self.Ny - 1:
                inte = (self.x_pos[1] - self.x_pos[0]) / (self.y_pos[-1] - self.y_pos[-2]) / 2.0
            else:
                inte = (self.x_pos[1] - self.x_pos[0]) / (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
                inte += (self.x_pos[1] - self.x_pos[0]) / (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
        
        elif x_index == self.Nx - 1:
            if y_index == self.Ny - 1:
                inte = (self.x_pos[-1] - self.x_pos[-2]) / (self.y_pos[-1] - self.y_pos[-2]) / 2.0
            elif y_index == 0:
                inte = (self.x_pos[-1] - self.x_pos[-2]) / (self.y_pos[1] - self.y_pos[0]) / 2.0
            else:
                inte = (self.x_pos[-1] - self.x_pos[-2]) / (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
                inte += (self.x_pos[-1] - self.x_pos[-2]) / (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0

        elif y_index == 0:
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1]) / (self.y_pos[1] - self.y_pos[0]) / 2.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) / (self.y_pos[1] - self.y_pos[0]) / 2.0

        elif y_index == self.Ny - 1:
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1]) / (self.y_pos[-1] - self.y_pos[-2]) / 2.0
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) / (self.y_pos[-1] - self.y_pos[-2]) / 2.0

        return inte

    def compute_product_basis_element(self, indices1: tuple, indices2: tuple):
        """Computes elements of the matrix for \int \psi_i \psi_j."""
        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, y_index1)):
            return 0 # for linear elements non-neighbors have 0 overlap
        
        if indices1 == indices2:
            return self.compute_integral_of_basis_function(x_index1, y_index1)
    
        return 1  # TODO: Implement full calculation

    def compute_product_basis_element_dx(self, indices1: tuple, indices2: tuple):
        """Computes matrix elements for the first derivative in X."""
        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        inte = 0

        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, y_index1)):
            return 0

        elif x_index2 == x_index1 + 1: # neighbor on the right
            if self.check_in_grid(x_index1+1, y_index1+1):
                inte = -1/(self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 2.0
            if self.check_in_grid(x_index1+1, y_index1-1):
                inte = inte - 1/(self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 2.0

        elif x_index2 == x_index1 - 1: # neighbor on the left
            if self.check_in_grid(x_index1-1, y_index1+1):
                inte = -1/(self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 2.0
            if self.check_in_grid(x_index1-1, y_index1-1):
                inte = inte - 1/(self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 2.0

        return  inte
    
    def compute_product_basis_element_dy(self, indices1: tuple, indices2: tuple):
        """Computes matrix elements for the first derivative in X."""
        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, y_index1)):
            return 0
    
        return 0  # TODO: Implement full calculation



# TO DO: USE SPARSE MATRICES TO BUILD MATRICES THAT WILL GO INTO FINAL EQUATION
# ONE SHOULD ACCEPT A GENERAL EQUATION, WITH GIVEN COEFFICIENTS FOR THE VARIOUS OPERATORS
# PER LE DERIVATE DIFFERENZIARE X E Y (CHECK)

# IDEA LUNGO TERMINE : INCLUDERE IDEE CORSO CNAM TIPO PINN. MA PRIMA FARE TEST DELLA CORRETTEZZA DELLA MESH
# SCRIVERE OUTPUT IN FILE LOG
# ALLA FINE CURARE ESTETICA DEL CODICE

# names : mass matrix and stiffness matrix (derivatives)
# termini con derivate prime: advection equation; va stabilizzata