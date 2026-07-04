# A 2d demonstration of the finite element method. And a demonstration of PINNs, applied to the heat equation

The Mesh class creates a 2d finite element structured mesh, with linear functions. 

The main limitation is that in order to generate a mesh one has to give a set of nodes in x-direction and a set of nodes in y-direction. It will not be possible, for example, to have a finer mesh in x-direction only in a given y range. This choice was taken for simplicity's sake. Future versions of this code might work with less uniform meshes.

The PINNs are built with PyTorch. The file PINN-3d.py contains the definition of two PINN classes: one for solving a 2d Poisson problem (PINN_Poisson_2d) and one for a 2+1 d heat equation, called PINN_heat_2d (evolution in time of a 2d temperature field). I use these classes in order to test them in the notebook pinn_trials.ipynb, where I also compare their outputs to the results obtained with the finite element algorithm.
I am currently testing various methods for choosing the collocation points and training the heat PINN, so far I have used LHS and RAR-G (see for instance https://arxiv.org/pdf/2207.10289). RAD is in the works. More details in the notebook.

# A quick overview of the maths : building the mass and stiffness matrices.

We now compute the matrix elements of the mass matrix $\int \psi_i \psi_j$

First, let's write down the involved functions.
Two triangles are involved. On the upper triangle we have


$$f(x, y) = 1 - \frac{x - x_{N}}{x_{N+1} - x_{N}}  - \frac{y - y_{N}}{y_{N+1} - y_{N}}$$
$$g(x, y) = 1 - \frac{x - x_{N+1}}{x_{N} - x_{N+1}} $$

The superposition integral is
$$I = \int_{x_N}^{x_{N+1}} \int_{y_N}^{y_{N+1} - \frac{x - x_{N}}{x_{N+1} - x_{N}}(y_{N+1} - y_{N})} \, f(x, y) g(x, y) \, \text{d}y \,\text{d}x$$

For simplicity we can set $x_N = y_N = 0$, $x_{N+1} = a$, $y_{N+1} = b$ in order to compute the integral.
In order to compute the diagonal elements $\int \psi_i \psi_i$, we have to evaluate the following integrals:
- 90 degrees angle from central node
$$\int_{x_N}^{x_{N+1}} \int_{y_N}^{y_{N+1} - \frac{x - x_{N}}{x_{N+1} - x_{N}}(y_{N+1} - y_{N})} \, f(x, y)^2 \, \text{d}y \,\text{d}x =  \frac{ab}{12}$$
with the same simplifications as above.
- linear decrease along $y$: also $\frac{ab}{12}$.
- linear decrease along $x$: also $\frac{ab}{12}$.

Note that $a$ and $b$ can vary if the mesh is not isotropic, so that we have to look for their values for every triangle that we consider. Sometimes we have two triangles lying next to eachother, so that we can add up their contributions and get $\frac{ab}{6}$.

Now let us compute the overlap integrals of the type $\int \psi_i \psi_j$.
- $j$ is the neighbor on the right of $i$. Triangle 2 gives $ab/24$, triangle 3 gives $ab/24$
- $j$ is the neighbor on the bottom right of $i$. Triangle 3 gives $ab/24$, triangle 4 gives $ab/24$
- and so on for the other triangles. The contribution is always $A/24$, where $A$ is the area of the triangle.

For the stiffness matrix $\int \nabla \psi_i \nabla \psi_j$ we perform similar calculations.

We set up the matrices as sparse matrices for better efficiency.

# How to use this code - Poisson equation
An example is provided in the file trials.py.

For the moment we can simulate the Poisson equation with Dirichlet boundary conditions. The function "run_simulation_poisson_dirichlet" takes the source function and the function giving the boundary conditions as arguments.

- One first defines the grid in x and y directions (with two separate arrays)
- One then initializes the mesh with the constructor Mesh(x, y), e.g. mymesh = Mesh(x, y)
- One then computes the solution with mymesh.run_simulation_poisson_dirichlet(func, diri)
- The solution is given as a 2d array, ready for plotting (for example as a colormap).

# How to use this code - Heat equation
SECTION TO BE COMPLETED


Note that, with explicit methods, for stability we need a high number of timesteps. The requirement seems to be well above the one posed by the CFL condition.

That is why in the notebook testing PINNs I use the implicit middle-point rule, which proves to be much more cost effective.