# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import sys
import hashlib
import numpy as np

EPSILON = sys.float_info.epsilon

class Differentiable(object):

    def __init__(self, parents=[]):
        self._value = None
        self._grad  = {}
        self._parents = []
        for parent in parents:
            if isinstance(parent, Differentiable):
                parent.add_child(self)
                self._parents.append(parent)
        self._children = []

    def value(self, rng=None, inputs=None):
        """Compute the value of the function.  This walks up the
        dependency graph and finds all of the Kayak objects with known
        values (such as Inputs and Targets, perhaps modulated by a
        Batcher) and then propagates their values forward through the
        modular computations of Differentiable subclasses.  There are
        some subtleties to this behavior, which are determined by the
        method arguments.

        Arguments:

          reset: (Boolean) Determines whether or not to use the cached
                 values in the graph.  If reset=True, then everything
                 will be recomputed from scratch.  If reset=False,
                 then any cached values will be reused.  Most of the
                 time, you'll want to use reset=True; the caching is
                 most useful when compute gradients on the backward
                 pass.

          rng: (None or numpy.random.RandomState) Some Kayak modules
               consume randomness, e.g., the Dropout module needs to
               compute a random mask; this argument allows you to pass
               a random number generator state down the chain for
               reproducability.  If rng=None, each module creates one
               as necessary.

          inputs: (Dictionary with Kayak objects as keys and numpy
                  arrays as values) After you do all the training with
                  Batcher objects and so on, you'll want to use the
                  function to actually compute something on novel
                  inputs.  You do that by giving this method a
                  dictionary that fixes values for various modules.
                  That is, when value() is called on a Kayak object
                  that appears as a key in the inputs dictionary, it
                  will immediately return the associated value in the
                  dictionary.

        """

        # # No need to recurse at all if this object's value is already determined.
        if inputs is not None and inputs.has_key(self):
            return inputs[self]

        # If we're resetting, or if the value is not yet cached, compute it.
        if self._value is None:
            self._value = self.compute_value(rng, inputs)

        return self._value

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
        return other.d_out_d_self(self)

    def d_out_d_self(self, out):
        if out in self._grad:
            return self._grad[out]

        if self is out:
            grad = 1.0
        elif len(self._children) == 0:
            grad = np.zeros(self.shape())
        else:
            grad = sum([child.d_out_d_parent(out, self) for child in self._children])

        self._grad[out] = grad
        return grad

    def d_out_d_parent(self, out, parent):
        assert parent in self._parents
        return self.local_grad(parent, self.d_out_d_self(out))

    def clear_value(self):
        """Recursively clears cached value and gradient by maintaining
        the invariant that if a node's _value is None that node's
        descendants' values and gradients are also None (or {}). This
        is the logical dual of saying that if a node does have a
        value, so must its parents."""

        if self._value is None:
            # Node is already clear
            return

        self.clear_grad()

        for child in self._children:
            child.clear_value()

        self._value = None

    def clear_grad(self):
        """Recursively clears gradient and those gradients that
        explicitly depend on it via backprop. If a node's _grad is
        empty, the node's parents' _grad variables are also empty."""
        if not self._grad:
            return

        for parent in self._parents:
            parent.clear_grad()

        self._grad = {}

    def local_grad(self, parent, d_out_d_self):
        """Return d_out_d_self * d_self_d_parent"""
        raise Exception("Class 'Differentiable' is abstract.")

    def add_child(self, child):
        """We need to keep track of our children."""
        self._children.append(child)

    def compute_value(self, rng, inputs):
        raise Exception("Class 'Differentiable' is abstract.")

    def shape(self, inputs=None):
        raise Exception("Class 'Differentiable' is abstract.")

from constants      import Constant, Parameter
from batcher        import Batcher
from inputs         import Inputs
from targets        import Targets
from matrix_ops     import MatAdd, MatMult, MatSum, Transpose, Reshape, Concatenate
from elem_ops       import ElemAdd
from nonlinearities import SoftReLU, HardReLU, LogSoftMax
from losses         import L2Loss, LogMultinomialLoss
from dropout        import Dropout
from regularizers   import L2Norm, L1Norm, Horseshoe, NExp
from crossval       import CrossValidator
