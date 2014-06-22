
class Inputs(object):

    def __init__(self, X, batcher=None):
        self.data  = X # Assume NxD

    def value(self, reset=False):
        # FIXME: batcher
        return self.data
    
    def shape(self):
        # FIXME: batcher
        return self.data.shape

    def grad(self, other):
        raise Exception("Not sensible to compute gradient in terms of inputs.")

    def depends(self, other):
        pass
