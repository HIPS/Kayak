import numpy as np
from .        import Differentiable

class Blank(Differentiable):
    # Creates a custom kayak node on-the-fly with compute_grad and/or local_grad
    # functions passed in as arguments
    def __init__(self, args=[], compute_value=None, local_grad=None):
        super(Blank, self).__init__(args)
        self.compute_value_fun = compute_value
        self.local_grad_fun = local_grad
    
    def _compute_value(self):
        return self.compute_value_fun(self._parents)

    def _local_grad(self, parent, d_out_d_self):
        return self.local_grad_fun(self._parents, parent, d_out_d_self)
