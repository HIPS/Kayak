
class Inputs(object):

    def __init__(self, X, batcher=None):
        self.data    = X # Assume NxD
        self.batcher = batcher

    def value(self, reset=False):
        if self.batcher is None:
            return self.data
        else:
            return self.data[self.batcher.indices(),:]
    
    def shape(self):
        if self.batcher is None:
            return self.data.shape
        else:
            return self.data[self.batcher.indices(),:].shape

    def grad(self, other):
        raise Exception("Not sensible to compute gradient in terms of inputs.")

    def depends(self, other):
        return False
