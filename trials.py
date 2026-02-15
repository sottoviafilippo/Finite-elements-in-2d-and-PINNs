import numpy as np
from FFEM_building_blocks import Mesh

x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 4)

mymesh = Mesh(x, y)

print(mymesh.grid)

print(mymesh.compute_integral_of_basis_function(1,1))
print(mymesh.compute_integral_of_basis_function(0,0))
print(mymesh.compute_integral_of_basis_function(0,1))
