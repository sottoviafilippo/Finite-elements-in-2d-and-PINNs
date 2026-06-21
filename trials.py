import numpy as np
from FFEM_building_blocks import Mesh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
# from numpy.linalg import norm
from matplotlib.animation import FFMpegWriter, FuncAnimation



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
plt.savefig('results/poisson.png')


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


# application: the heat equation. (with Dirichlet boundary conditions)
# consider a square geometry. the edges are kept at a constant temperature of 100 °C.
# the starting temperature is 0 °C inside the surface

Nx = Ny = 100

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)

# initialize the starting temperature (20) and the boundary temperature (100)
u0 = 20.0 + np.zeros(Nx*Ny)
for i in [0, Nx-1]:
    for j in range(0, Ny):
                u0[Ny*i + j] = 100
        
for j in [0, Ny-1]:
    for i in range(1, Nx - 1):
                u0[Ny*i + j] = 100

u = u0.copy()

# plot initial conditions for a check
plt.figure(figsize=(12, 6))
plt.imshow(u.reshape((Nx, Ny)).transpose(), cmap='viridis', origin='lower', aspect='auto', extent=[x.min(), x.max(), y.min(), y.max()]) # need origin "lower"
plt.colorbar()
plt.title('Poisson equation result - Dirichlet b.c.')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('results/initial_conditions.png')

mymesh = Mesh(x, y, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()

steps = 1000
times, dt = np.linspace(0, 0.1, steps, retstep = True)

fig, ax = plt.subplots()
im = ax.imshow(u.reshape((Nx, Ny)).transpose(), cmap='magma', animated=True, interpolation='bilinear')
ax.set_title("2D Heat Equation Evolution")
plt.colorbar(im, label='Temperature')

def update(frame):
    
    # first approach: explicit Euler in time for simplicity
    # use the dedicated function defined in the Mesh class
    global u
    unew = u.copy()
    unew = unew + dt * mymesh.compute_time_derivative_heat_equation(unew, alpha = 0.01)
    # CFL condition dt <= 1/4(alpha). In practice one needs to be more restrictive

    # impose b.c. at every time (constant temperature on the edge)
    for i in [0, Nx-1]:
        for j in range(0, Ny):
                unew[Ny*i + j] = 100
    for j in [0, Ny-1]:
        for i in range(1, Nx - 1):
                unew[Ny*i + j] = 100

    u = unew
    im.set_array(u.reshape((Nx, Ny)).transpose()) # Update the image data for this frame
    return [im]

# now produce the animation
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
# save the animation in mp4 format
ani.save("results/heat_equation.mp4", writer=writer)

plt.show()
