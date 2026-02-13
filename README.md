# A 2d demonstration of the finite element method

The Mesh class creates a 2d finite element structured mesh, with linear elements. 

# Intersection of the basis functions $f$ and $g$ respectively for two nodes $(x_N, y_N)$ and $(x_{N+1}, y_N)$ with the same y coordinate. 

The basis functions are not normalized yet.
Two triangles are involved. On the upper triangle we have


$$f(x, y) = 1 - \frac{x - x_{N}}{x_{N+1} - x_{N}}  - \frac{y - y_{N}}{y_{N+1} - y_{N}}$$
$$g(x, y) = 1 - \frac{x - x_{N+1}}{x_{N} - x_{N+1}} $$

The superposition integral is
$$\int_{x_N}^{x_{N+1}} \int_{y_N}^{y_{N+1} - \frac{x - x_{N}}{x_{N+1} - x_{N}}(y_{N+1} - y_{N})} \, f(x, y) g(x, y) \, \text{d}y \,\text{d}x$$

Similarly, in order to compute the normalizations of the basis functions, we have to evaluate the following integral:
$$\int_{x_N}^{x_{N+1}} \int_{y_N}^{y_{N+1} - \frac{x - x_{N}}{x_{N+1} - x_{N}}(y_{N+1} - y_{N})} \, f(x, y)^2 \, \text{d}y \,\text{d}x$$