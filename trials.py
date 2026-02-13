import numpy as np
from FFEM_building_blocks import Mesh

x = [1,2,3]
y = [4,5,6]

mymesh = Mesh(x, y)

print(mymesh.grid)
