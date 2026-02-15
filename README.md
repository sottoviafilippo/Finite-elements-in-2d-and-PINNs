# A 2d demonstration of the finite element method

The Mesh class creates a 2d finite element structured mesh, with linear functions. 

The main limitation is that in order to generate a mesh one has to give a set of nodes in x-direction and a set of nodes in y-direction. It will not be possible, for example, to have a finer mesh in x-direction only in a given y range. This choice was taken for simplicity's sake. Future versions of this code might work with less uniform meshes.

# Intersection of the basis functions $f$ and $g$ respectively for two nodes $(x_N, y_N)$ and $(x_{N+1}, y_N)$ with the same y coordinate. 

We now compute the matrix elements $\int \psi_i \psi_j$

First, let's write down the functions.
Two triangles are involved. On the upper triangle we have


$$f(x, y) = 1 - \frac{x - x_{N}}{x_{N+1} - x_{N}}  - \frac{y - y_{N}}{y_{N+1} - y_{N}}$$
$$g(x, y) = 1 - \frac{x - x_{N+1}}{x_{N} - x_{N+1}} $$

The superposition integral is
$$I = \int_{x_N}^{x_{N+1}} \int_{y_N}^{y_{N+1} - \frac{x - x_{N}}{x_{N+1} - x_{N}}(y_{N+1} - y_{N})} \, f(x, y) g(x, y) \, \text{d}y \,\text{d}x$$

For simplicity we can set $x_N = y_N = 0$, $x_{N+1} = a$, $y_{N+1} = b$ in order to compute the integral.
We get:
$$I = \int_0^a \int_0^{b(1 - \frac{x }{a})} \, \left(1 - \frac{x}{a} - \frac{y}{b}\right)\left(1 - \frac{x}{a}\right) \, \text{d}y \,\text{d}x = \frac{ab}{8}$$


Similarly, in order to compute the diagonal elements $\int \psi_i \psi_i$, we have to evaluate the following integrals:
- 90 degrees angle from central node
$$\int_{x_N}^{x_{N+1}} \int_{y_N}^{y_{N+1} - \frac{x - x_{N}}{x_{N+1} - x_{N}}(y_{N+1} - y_{N})} \, f(x, y)^2 \, \text{d}y \,\text{d}x =  \frac{ab}{12}$$
with the same simplifications as above.
- linear decrease along $y$: also $\frac{ab}{12}$.
- linear decrease along $x$: also $\frac{ab}{12}$.

Note that $a$ and $b$ can vary if the mesh is not isotropic, so that we have to look for their values for every triangle that we consider. Sometimes we have two triangles lying next to eachother, so that we can add up their contributions and get $\frac{ab}{6}$.