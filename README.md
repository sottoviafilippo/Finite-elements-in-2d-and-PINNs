# A 2d demonstration of the finite element method

The Mesh class creates a 2d finite element grid, with linear elements. 

# Intersection of the basis functions $f$ and $g$ respectively for two nodes $(x_N, y_N)$ and $(x_{N+1}, y_N)$ with the same y coordinate. 

The basis functions are not normalized yet.
Two triangles are involved. On the upper triangle we have


$$f = 1 - \frac{x}{x_{N+1} - x_{N}}  - \frac{y}{y_{N+1} - y_{N}}$$
$$g = (x - x_N)(y - y_N)$$