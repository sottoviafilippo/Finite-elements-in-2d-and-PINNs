import numpy as np
import warnings
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

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
            [x_index - 1, y_index + 1], 
            [x_index, y_index + 1], 
            [x_index + 1, y_index], 
            [x_index, y_index - 1],
            [x_index - 1, y_index - 1]
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
        if not self.check_in_grid(x_index, y_index):
            warnings.warn("Warning: indices out of range", UserWarning)
            return 0

        inte = 0.0  # Initialize integral value

        if self.check_in_grid(x_index - 1, y_index + 1):
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 6.0
        if self.check_in_grid(x_index + 1, y_index + 1):
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 12.0
        if self.check_in_grid(x_index + 1, y_index - 1):
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 6.0
        if self.check_in_grid(x_index - 1, y_index - 1):
            inte += (self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 12.0

        return inte
    
    def compute_integral_of_basis_function_dx(self, x_index: int, y_index: int):
        """Computes the diagonal elements of \int d_x\psi_i d_x\psi_j."""
        if not self.check_in_grid(x_index, y_index):
            warnings.warn("Warning: indices out of range", UserWarning)
            return 0

        inte = 0.0  # Initialize integral value

        if self.check_in_grid(x_index + 1, y_index + 1):
            inte += 1/(self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
        if self.check_in_grid(x_index + 1, y_index - 1):
            inte += 1/(self.x_pos[x_index + 1] - self.x_pos[x_index]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
        if self.check_in_grid(x_index - 1, y_index - 1):
            inte += 1/(self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
        if self.check_in_grid(x_index - 1, y_index + 1):
            inte += 1/(self.x_pos[x_index] - self.x_pos[x_index - 1]) * (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0

        return inte

    def compute_integral_of_basis_function_dy(self, x_index: int, y_index: int):
        """Computes the diagonal elements of \int d_y\psi_i d_y\psi_j."""
        if not self.check_in_grid(x_index, y_index):
            warnings.warn("Warning: indices out of range", UserWarning)
            return 0

        inte = 0.0  # Initialize integral value

        if self.check_in_grid(x_index-1, y_index+1):
            inte += (self.x_pos[x_index] - self.x_pos[x_index - 1]) / (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
        if self.check_in_grid(x_index+1, y_index+1):
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) / (self.y_pos[y_index + 1] - self.y_pos[y_index]) / 2.0
        if self.check_in_grid(x_index+1, y_index-1):
            inte += (self.x_pos[x_index + 1] - self.x_pos[x_index]) / (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0
        if self.check_in_grid(x_index-1, y_index-1):
            inte += (self.x_pos[x_index] - self.x_pos[x_index - 1]) / (self.y_pos[y_index] - self.y_pos[y_index - 1]) / 2.0

        return inte

    def compute_product_basis_element(self, indices1: tuple, indices2: tuple):
        """Computes elements of the matrix for \int \psi_i \psi_j."""
        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        if indices1 == indices2:
            return self.compute_integral_of_basis_function(x_index1, y_index1)

        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, y_index1)):
            return 0 # for linear elements non-neighbors have 0 overlap
        
        inte = 0.0

        if x_index2 == x_index1 + 1 and y_index2 == y_index1: # neighbor on the right
            if self.check_in_grid(x_index1+1, y_index1+1):
                inte = (self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 24.0
            if self.check_in_grid(x_index1+1, y_index1-1):
                inte = inte + (self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 24.0

        elif x_index2 == x_index1 + 1 and y_index2 == y_index1 - 1: # neighbor on the bottom right
            inte = (self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 12.0

        elif x_index2 == x_index1 and y_index2 == y_index1 - 1: # neighbor on the bottom
            if self.check_in_grid(x_index1+1, y_index1-1):
                inte = (self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 24.0
            if self.check_in_grid(x_index1-1, y_index1-1):
                inte = inte + (self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 24.0

        elif x_index2 == x_index1 - 1 and y_index2 == y_index1: # neighbor on the left
            if self.check_in_grid(x_index1-1, y_index1-1):
                inte = (self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 24.0
            if self.check_in_grid(x_index1-1, y_index1+1):
                inte = inte + (self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 24.0
        
        elif x_index2 == x_index1 - 1 and y_index2 == y_index1 + 1: # neighbor on the top left
            inte = (self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 12.0

        elif x_index2 == x_index1 and y_index2 == y_index1 + 1: # neighbor on the top
            if self.check_in_grid(x_index1-1, y_index1+1):
                inte = (self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 24.0
            if self.check_in_grid(x_index1+1, y_index1+1):
                inte = inte + (self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 24.0

        return inte  

    def compute_product_basis_element_dx(self, indices1: tuple, indices2: tuple):
        """Computes matrix elements for the first derivative in X."""
        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        if indices1 == indices2:
            return self.compute_integral_of_basis_function_dx(x_index1, y_index1)

        inte = 0.0

        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, y_index1)):
            return 0

        elif x_index2 == x_index1 + 1 and y_index2 == y_index1: # neighbor on the right
            if self.check_in_grid(x_index1+1, y_index1+1):
                inte = -1/(self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 2.0
            if self.check_in_grid(x_index1+1, y_index1-1):
                inte = inte - 1/(self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 2.0

        elif x_index2 == x_index1 - 1 and y_index2 == y_index1: # neighbor on the left
            if self.check_in_grid(x_index1-1, y_index1+1):
                inte = -1/(self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 2.0
            if self.check_in_grid(x_index1-1, y_index1-1):
                inte = inte - 1/(self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) * (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 2.0

        # top left and bottom right neighbors give 0

        return  inte
    
    def compute_product_basis_element_dy(self, indices1: tuple, indices2: tuple):
        """Computes matrix elements for the first derivative in X."""
        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        if indices1 == indices2:
            return self.compute_integral_of_basis_function_dy(x_index1, y_index1)

        inte = 0.0

        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, y_index1)):
            return 0
    
        elif y_index2 == y_index1 + 1: # neighbor above
            if self.check_in_grid(x_index1-1, y_index1+1):
                inte = -(self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) / (self.y_pos[y_index1 + 1] - self.y_pos[y_index1]) / 2.0
            if self.check_in_grid(x_index1+1, y_index1+1):
                inte = inte - (self.x_pos[x_index1 + 1] - self.x_pos[x_index1]) / (self.y_pos[y_index1+1] - self.y_pos[y_index1]) / 2.0

        elif y_index2 == y_index1 - 1: # neighbor below
            if self.check_in_grid(x_index1-1, y_index1-1):
                inte = -(self.x_pos[x_index1] - self.x_pos[x_index1 - 1]) / (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 2.0
            if self.check_in_grid(x_index1+1, y_index1-1):
                inte = inte - (self.x_pos[x_index1+1] - self.x_pos[x_index1]) / (self.y_pos[y_index1] - self.y_pos[y_index1 - 1]) / 2.0

        return inte  

    def build_mass_matrix(self, verbose = True):
        """ builds the mass matrix using the functions defined above. Using sparse matrices"""

        #order: x0, y0 ... x0, yN ... x1,y0 ... etc
        N = self.Nx * self.Ny
        M = lil_matrix((N, N))

        for i in range(0, self.Nx):
            for j in range(0, self.Ny):
                total_index = self.Ny*i + j
                
                M[total_index, total_index] = self.compute_integral_of_basis_function(i, j)
                neighbors = self.find_neighbors(i, j)

                for neighbor in neighbors:
                    total_index_neighbor = self.Ny*neighbor[0] + neighbor[1]
                
                    M[total_index, total_index_neighbor] = self.compute_product_basis_element([i, j], neighbor)

        self.M = M

        if verbose:
            print("Mass matrix initialization completed")



# IDEA LUNGO TERMINE : INCLUDERE IDEE CORSO CNAM TIPO PINN. MA PRIMA FARE TEST DELLA CORRETTEZZA DELLA MESH
# SCRIVERE OUTPUT IN FILE LOG
# ALLA FINE CURARE ESTETICA DEL CODICE

# names : mass matrix and stiffness matrix (derivatives)
# termini con derivate prime: advection equation; va stabilizzata
# should speed up M matrix initialization by considering symmetry