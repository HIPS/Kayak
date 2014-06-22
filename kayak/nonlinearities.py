import numpy as np

from . import Differentiable

class ReLU(Differentiable):

    def __init__(self, A, scale=1.0):
        self.A     = A
        self.scale = scale

    def eval(self):
        pass

    def value(self):
        return np.log(1.0 + np.exp( self.A.value() / self.scale )) * self.scale

    def grad(self, other):
        if other == self.A:
            return Logistic( Scale( self.A, 1.0/self.scale ))
        elif self.A.depends(other):
            return ElemMult( self.A.grad(other), Logistic( Scale( self.A, 1.0/self.scale )))
        else:
            return Zeros(other.shape)

    def depends(self, other):
        return self.A == other or self.A.depends(other)

    def shape(self):
        return self.A.shape()

class Logistic(Differentiable):
    
    def __init__(self, A):
        self.A = A
    
    def eval(self):
        pass

    def value(self):
        return 1.0 / (1.0 + np.exp( -self.A.value()))

    def grad(self, other):
        if other == self.A:
            return ElemMult( Logistic(self.A), Logistic(Scale(self.A, -1.0)) )
        elif self.A.depends(other):
            return ElemMult( self.A.grad(other), ElemMult( Logistic(self.A), Logistic(Scale(self.A, -1.0)) ))
        else:
            return Zeros(other.shape)

    def shape(self):
        return self.A.shape()
