import numpy as np

class Differentiable(object):

    def __init__(self, parents=[]):
        self._value = None
        self._grad  = {}
        for parent_index, parent in enumerate(parents):
            parent._add_child(self, parent_index)

        self._parents = parents
        self._children = []

    @property
    def value(self):
        """Compute the value of the function.  This walks up the
        dependency graph and finds all of the Kayak objects with known
        values (such as Inputs and Targets, perhaps modulated by a
        Batcher) and then propagates their values forward through the
        modular computations of Differentiable subclasses.  There are
        some subtleties to this behavior, which are determined by the
        method arguments.
        """
        # If the value is not yet cached, compute it.
        if self._value is None:
            self._value = self._compute_value()

        return self._value

    @value.setter
    def value(self, new_value):
        self._clear_value_cache()
        self._value = new_value

    def _clear_value_cache(self):
        """
        Sets the new value and clears the values of any dependencies. We
        maintain the invariant that cached values are never wrong relative
        to their parents' values.
        """
        if self._value is not None:
            [child._clear_value_cache() for child, _ in self._children]
            self._clear_grad_cache()
            self._value = None

    def _clear_grad_cache(self):
        if self._grad:
            [parent._clear_grad_cache() for parent in self._parents]
            self._grad = {}
        
    def grad(self, other):
        """Compute the gradient of this module in terms of another
        module.  One of the main points of the Kayak setup is to
        easily compute gradients in terms of parameters.  This is the
        interface for doing so.  You call the grad() method on
        something that produces a scalar, providing as an argument
        some other object that appears deeper in the graph.  You get
        out an array of the same shape as the deeper object, but which
        is the gradient.

        Arguments:

          other: (Kayak object) The other object, in terms of which
                 you'd like to take this thing's gradient.
        """
        return other._d_out_d_self(self)

    @property
    def shape(self):
        return self.value.shape

    def _d_out_d_self(self, out):
        if out not in self._grad:
            if self is out:
                grad = np.ones(self.shape)
            elif not self._children:
                grad = 0
            else:
                grad = sum(child._d_out_d_parent(out, parent_index)
                           for child, parent_index in self._children)
            self._grad[out] = grad

        return self._grad[out]

    def _d_out_d_parent(self, out, parent):
        d_out_d_self = self._d_out_d_self(out)
        if d_out_d_self is 0:
            # This avoid calling local_grad for paths that don't end in 'out'
            return 0
        else:
            return self._local_grad(parent, d_out_d_self)

    def _add_child(self, child, parent_index):
        """Parent_index is an int that tells out child which parent we are."""
        self._children.append((child, parent_index))

    def _local_grad(self, parent, d_out_d_self):
        """Return d_out_d_self * d_self_d_parent"""
        raise Exception("Class 'Differentiable' is abstract.")

    def _compute_value(self):
        raise Exception("Class 'Differentiable' is abstract.")

    # Overload plus and times operators with elementwise operations
    # To avoid circular imports, we wait until the operator is called
    # to import the subclasses of Differentiable
    def __add__(self, other):
        from . import ElemAdd, Constant

        # If other is not a Differentiable object,
        # try to cast it as a constant.
        if not isinstance(other, Differentiable):
            other = Constant(other)
        return ElemAdd(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        from . import ElemMult, Constant
        # If other is not a Differentiable object,
        # try to cast it as a constant.
        if not isinstance(other, Differentiable):
            other = Constant(other)
        return ElemMult(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # NOTE: Assuming Python 2.x syntax for div
    def __div__(self, other):
        from . import ElemPower
        return self * ElemPower(other, -1)

    def __rdiv__(self, other):
        from . import ElemPower
        return other * ElemPower(self, -1)

    def __neg__(self):
        from . import ElemMult, Constant
        return ElemMult(Constant(-1.), self)

    def __pow__(self, power, modulo=None):
        from . import ElemPower
        return ElemPower(self, power)

    def __abs__(self):
        from . import ElemAbs
        return ElemAbs(self)



