import numpy as np
import warnings

class Mesh:
    """ Produces a 2D finite element computational grid and computes the associated matrices for the resolution of differential equations"""
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
        
        if x_index >= self.Nx or y_index >= self.Ny:
            warnings.warn("Warning: index out of range", UserWarning)
            return 0
        
        all_possible_neighbors = [[x_index - 1, y_index], [x_index, y_index + 1], [x_index + 1, y_index], [x_index, y_index - 1]]

        """ only return elements within the grid """
        return [nb for nb in all_possible_neighbors if nb[0] >= 0 and nb[1] >= 0 and nb[0] <= self.Nx-1 and nb[1] <= self.Ny-1]
    

    def compute_integral_of_basis_function(self, x_index: int, y_index: int):
        
        """ computes the diagonal elements of \int \psi_i \psi_j """

        if x_index < 0 or y_index < 0:
            warnings.warn("Warning: negative index", UserWarning)
            return 0
        
        if x_index >= self.Nx or y_index >= self.Ny:
            warnings.warn("Warning: index out of range", UserWarning)
            return 0

        n_nb = len(self.find_neighbors(x_index, y_index)) # number of neighbors

        """ first the case of elements within the grid, i.e. with 4 neighbors """
        inte = 0 # here we will store the value of the integral, for the moment we just initialize it to 0 for clarity and debugging

        if n_nb == 4:
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1])*(self.y_pos[y_index + 1] - self.y_pos[y_index ])/6.
            inte = inte + (self.x_pos[x_index + 1] - self.x_pos[x_index])*(self.y_pos[y_index + 1] - self.y_pos[y_index ])/12.
            inte = inte + (self.x_pos[x_index + 1] - self.x_pos[x_index])*(self.y_pos[y_index] - self.y_pos[y_index - 1])/6.
            inte = inte + (self.x_pos[x_index] - self.x_pos[x_index - 1])*(self.y_pos[y_index ] - self.y_pos[y_index - 1])/12.

        elif x_index == 0: 
            if y_index == 0: # one single triangle
                inte = (self.x_pos[1] - self.x_pos[0])*(self.y_pos[1] - self.y_pos[0])/12.

            elif y_index == self.Ny - 1: # two triangles
                inte = (self.x_pos[1] - self.x_pos[0])*(self.y_pos[-1] - self.y_pos[-2])/6.

            elif y_index != 0 and y_index != self.Ny-1: # three triangles
                inte = (self.x_pos[1] - self.x_pos[0])*(self.y_pos[y_index + 1] - self.y_pos[y_index ])/12.
                inte = inte + (self.x_pos[1] - self.x_pos[0])*(self.y_pos[y_index] - self.y_pos[y_index - 1])/6.
        
        elif x_index == self.Nx - 1:
            if y_index == self.Ny - 1: # one single triangle
                inte = (self.x_pos[-1] - self.x_pos[-2])*(self.y_pos[-1] - self.y_pos[-2])/12.

            elif y_index == 0: # two triangles
                inte = (self.x_pos[-1] - self.x_pos[-2])*(self.y_pos[1] - self.y_pos[0])/6.

        elif x_index != 0 and x_index != self.Nx - 1 and y_index == 0: # three triangles
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1])*(self.y_pos[1] - self.y_pos[0])/6.
            inte = inte + (self.x_pos[x_index + 1] - self.x_pos[x_index])*(self.y_pos[1] - self.y_pos[0])/12.

        elif x_index == self.Nx - 1 and y_index != self.Ny - 1 and y_index !=0: # three triangles
            inte = (self.x_pos[-1] - self.x_pos[-2])*(self.y_pos[y_index] - self.y_pos[y_index - 1])/12.
            inte = inte + (self.x_pos[-1] - self.x_pos[-2])*(self.y_pos[y_index + 1] - self.y_pos[y_index])/6.

        elif x_index != self.Nx - 1 and x_index != 0 and y_index == self.Ny - 1: # three triangles
            inte = (self.x_pos[x_index] - self.x_pos[x_index - 1])*(self.y_pos[-1] - self.y_pos[-2])/12.
            inte = inte + (self.x_pos[x_index + 1] - self.x_pos[x_index])*(self.y_pos[-1] - self.y_pos[-2])/6.

        return inte

    def compute_product_basis_element(self, indices1, indices2):

        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        """ computes the elements of the matrix for the \int \psi_i \psi_j, given two sets of indices """

        """ if the two points are not neighbors, returns 0"""
        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, x_index2)):
            return 0
        
        if indices1 == indices2:
            """ in this case we use the function that we defined above """
            return self.compute_integral_of_basis_function(x_index1, y_index1)
    
        return 1 #TO BE FIXED AND COMPLETED    

    def compute_der_x_basis_element(self, indices1, indices2):

        x_index1, y_index1 = indices1
        x_index2, y_index2 = indices2

        """ computes the elements of the matrix for the first derivative, given two sets of indices """

        """ if the two points are not neighbors, returns 0"""
        if not ([x_index2, y_index2] in self.find_neighbors(x_index1, x_index2)):
            return 0
    
        return 1 #TO BE FIXED AND COMPLETED
    



# TO DO: USE SPARSE MATRICES TO BUILD MATRICES THAT WILL GO INTO FINAL EQUATION
# ONE SHOULD ACCEPT A GENERAL EQUATION, WITH GIVEN COEFFICIENTS FOR THE VARIOUS OPERATORS
# PER LE DERIVATE DIFFERENZIARE X E Y (CHECK)

# IDEA LUNGO TERMINE : INCLUDERE IDEE CORSO CNAM TIPO PINN. MA PRIMA FARE TEST DELLA CORRETTEZZA DELLA MESH
# SCRIVERE OUTPUT IN FILE LOG
# ALLA FINE CURARE ESTETICA DEL CODICE