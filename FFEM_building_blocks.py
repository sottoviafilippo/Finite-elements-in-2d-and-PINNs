import numpy as np

class Mesh:
    """ Produces a 2D finite element computational grid"""


    def __init__(self, x_positions, y_positions):
        """ initializes the class with the two positions arrays, creates the grid """

        self.x_pos = x_positions
        self.y_pos = y_positions
        self.grid  = np.meshgrid(x_positions, y_positions)

    