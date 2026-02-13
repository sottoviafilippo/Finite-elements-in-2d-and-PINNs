import numpy as np
import warnings

class Mesh:
    """ Produces a 2D finite element computational grid"""
    """ Triangular mesh with linear elements"""


    def __init__(self, x_positions: np.ndarray, y_positions: np.ndarray):
        """ initializes the class with the two positions arrays, creates the grid """

        self.x_pos = x_positions
        self.y_pos = y_positions
        self.grid  = np.meshgrid(x_positions, y_positions)

        """ now compute the nunmber of grid points in both directions """
        self.Nx    = len(self.x_pos)
        self.Ny    = len(self.y_pos)


    def find_neighbors(self, x_index: int, y_index: int):

        """ finds the list of neighbors of a given grid point """

        if x_index < 0 or y_index < 0:
            warnings.warn("Warning: negative index", UserWarning)
            return 0
        
        all_possible_neighbors = [[x_index + 1, y_index], [x_index - 1, y_index], [x_index, y_index + 1], [x_index, y_index - 1]]

        """ only return elements within the grid """
        return [nb for nb in all_possible_neighbors if nb[0] >= 0 and nb[1] >= 0 and nb[0] <= self.Nx-1 and nb[1] <= self.Ny-1]
    

    def compute_integral_of_basis_function(self, x_index: int, y_index: int)
        
        """ needed later to normalize the function """