import numpy as np
from FFEM_building_blocks import Mesh

x = [1,2,3,4]
y = [5,6,7,8]

mymesh = Mesh(x, y)

print(mymesh.compute_derivative_basis_element([0,0], [3,3]))
print(mymesh.compute_derivative_basis_element([0,0], [0,1]))
