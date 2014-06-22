
class Targets(object):

    def __init__(self, Y, batcher=None):
        self.data = Y # Assume NxP

    def value(self, reset=False):
        # FIXME batcher
        return self.data

    def shape(self):
        # FIXME batcher
        return self.data.shape

    def grad(self, other):
        raise Exception("Not sensible to take gradient in terms of targets.")

    def depends(self, other):
        pass
