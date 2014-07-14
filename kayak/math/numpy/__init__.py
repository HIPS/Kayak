import numpy        as np
import numpy.random as npr

class Matrix:

    def __init__(self, *shape):
        self._mat = np.zeros(*shape)

    
