# inspiration (and theory) from arXiv:2403.00599v1
# obviously redundant work. written by me to understand PINNs and how they are built from the ground up

import numpy as np
from numpy.random import uniform


class PINN_heat_3d:
    """
    PINN for 2+1d heat equation
    """

    # initialisation: see Glorot initilisation (good for tanh/sigmoid)
    # the biases b in the pass forward equation can be initialized to 0

    def __init__(self, N_layers: int, N_nodes: int, input_dim = 3, output_dim = 1):
        
        self.N_nodes = [3] + [N_nodes] * N_layers + [1] # number of nodes in each layer. The first layer corresponds to the entry data, the last one to the output

        self.weights = []
        for k in range(N_layers):
            a = np.sqrt(6./(self.N_nodes[k] + self.N_nodes[k+1]))# half-length of the interval from which we draw the uniform distribution
            self.weights.append(np.random.uniform(-a, a, size = N_nodes))
        
        pass
        
