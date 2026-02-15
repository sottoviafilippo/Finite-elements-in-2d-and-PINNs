import numpy as np
from FFEM_building_blocks import Mesh
from scipy.sparse import csr_matrix

x = np.linspace(0, 1, 3)
y = np.linspace(0, 1, 2)

mymesh = Mesh(x, y)
mymesh.build_mass_matrix()

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

print(is_sparse_symmetric(mymesh.M))
print(mymesh.M.toarray())


# TO DO: MATRICE NON SIMMETRICA PER COLPA DI (PARE) UN SOLO PUNTO, che sembra saltato. RISOLVERE