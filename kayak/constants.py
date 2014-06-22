from . import Differentiable


class Constant(Differentiable):

    def __init__(self, val):
        self._value = val

    def value(self, reset=False):
        return self._value

    def grad(self, other):
        return Zeros(other.shape())

    def depends(self, other):
        return self == other

    def shape(self):
        return self._value.shape

class Zeros(Constant):

    def __init__(self, shape):
        super(Zeros, self).__init__(np.zeros(shape))

class Parameter(Constant):

    def __init__(self, val):
        super(Parameter, self).__init__(val)

    def add(self, addend):
        self._value += addend
