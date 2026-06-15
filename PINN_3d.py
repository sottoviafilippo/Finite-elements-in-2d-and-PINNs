# inspiration (and theory) from arXiv:2403.00599v1
# obviously redundant work. written by me to understand PINNs and how they are built from the ground up

import numpy as np


class PINN_heat_3d:
    """
    PINN for 2+1d heat equation
    """

    # initialisation: see Glorot initilisation (good for tanh/sigmoid)
