import numpy as np

from . import Differentiable

class DataNode(Differentiable):

    def __init__(self, data, batcher=None):
        if batcher is None:
            super(DataNode, self).__init__([])
        else:
            super(DataNode, self).__init__([batcher])

        self.data    = np.atleast_1d(data)
        self.batcher = batcher

    def _compute_value(self):
        if self.batcher is None:
            return self.data
        else:
            return self.data[self.batcher.value,...]

    def _local_grad(self, parent, d_out_d_self):
        raise Exception("Can't take gradient w.r.t. data")

# TODO: Consider removing these
class Inputs(DataNode):
    def __init__(self, data, batcher=None):
        super(Inputs, self).__init__(data, batcher)
class Targets(DataNode):
    def __init__(self, data, batcher=None):
        super(Targets, self).__init__(data, batcher)
