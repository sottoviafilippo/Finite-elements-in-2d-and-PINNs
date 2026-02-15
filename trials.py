import numpy as np
from FFEM_building_blocks import Mesh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from numpy.linalg import norm

x = np.linspace(0, 4, 200)
y = np.linspace(0, 2, 100)

mymesh = Mesh(x, y, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()

# this function was used for tests
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

print(np.shape(res))

plt.figure(figsize=(12, 6))
plt.imshow(res, cmap='viridis', origin='lower', aspect='auto', extent=[x.min(), x.max(), y.min(), y.max()]) # need origin "lower"
plt.colorbar()
plt.title('Poisson equation result - Dirichlet b.c.')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()


"""
# now study the convergence

Ns = [2**k for k in np.arange(2, 9)]
relative_rests = np.zeros_like(Ns)

for k in range(len(Ns)):
    print(Ns[k])
    x = np.linspace(0, 1, Ns[k])
    y = np.linspace(0, 1, Ns[k])
    mymesh = Mesh(x, y, verbose=False)
    res = mymesh.run_simulation_poisson_dirichlet(func, diri)
    analytical_sol = np.zeros_like(res)
    for m in range(len(x)):
        for j in range(len(y)):
            analytical_sol[m, j] = np.sin(np.pi * x[m]) * np.sin(np.pi * y[j])

    print(norm(analytical_sol))
    relative_rests[k] = norm(analytical_sol.transpose() - res)/norm(analytical_sol)

print(relative_rests)

plt.figure()
plt.loglog(Ns, relative_rests)
plt.xlabel('$N$')
plt.ylabel('Relative rest')
plt.show()

#strangely it seems to produce a perfect solution??

"""