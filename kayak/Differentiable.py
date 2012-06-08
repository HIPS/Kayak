class Differentiable(object):

    def __init__(self, dims):
        raise Exception("Class 'Differentiable' is abstract.")

    def value(self):
        raise Exception("Class 'Differentiable' is abstract.")

    def gradient(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    #def hessian(self, other, vec):
    #    raise Exception("Class 'Differentiable' is abstract.")
