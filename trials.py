import numpy as np
from FFEM_building_blocks import Mesh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

mymesh = Mesh(x, y, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()

def is_sparse_symmetric(A: csr_matrix, tol: float = 1e-6) -> bool:
    """
    Checks if a sparse matrix is symmetric within a given tolerance. from Gemini
    """
    # Calculate the absolute difference between A and its transpose
    diff = A - A.transpose()
    
    # Check if the maximum absolute value in the difference is near zero
    # .data only looks at non-zero entries, making it very fast
    if diff.nnz == 0:
        return True
    
    return np.max(np.abs(diff.data)) < tol

func = lambda x,y: 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
diri = lambda x,y: 0

res = mymesh.run_simulation_poisson_dirichlet(func, diri)

plt.figure(figsize=(8, 6))
plt.imshow(res, cmap='viridis', origin='lower', aspect='auto', extent=[x.min(), x.max(), y.min(), y.max()]) # need origin "lower"
plt.colorbar()
plt.title('Poisson equation result - Dirichlet b.c.')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.show()
