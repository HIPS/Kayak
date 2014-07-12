# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import sys
import hashlib
import numpy as np

EPSILON = sys.float_info.epsilon

class Differentiable(object):

    def __init__(self):
        self._value = None
        self._grad  = {}

    def value(self, reset=False, rng=None, inputs=None):
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

        # No need to recurse at all if this object's value is already determined.
        if inputs is not None and inputs.has_key(self):
            return inputs[self]

        # If we're resetting, or if the value is not yet cached, compute it.
        if reset or self._value is None:
            self._value = self.compute_value(reset, rng, inputs)
            self._grad  = {} # Throw away old gradients.
        return self._value

    def grad(self, other, outgrad=1.0):
        """Compute the gradient of this module in terms of another
        module.  One of the main points of the Kayak setup is to
        easily compute gradients in terms of parameters.  This is the
        interface for doing so.  You call the grad() method on
        something that produces a scalar, providing as an argument
        some other object that appears deeper in the graph.  You get
        out an array of the same shape as the deeper object, but which
        is the gradient.  The trick is that there is an optional
        argument that allows you to do this with backpropagation, so
        if you hand it the upstream gradient, it multiplies it by the
        Jacobian on the way down the chain.

        Arguments:

          other: (Kayak object) The other object, in terms of which
                 you'd like to take this thing's gradient .

          outgrad: (float, numpy array, optional) The gradient of this
                   object's outputs, which you need to compute in
                   reverse mode.  That is, this is the bit being
                   backpropagated.

        """

        # We need distinct gradients for different things we might
        # want to differentiate in terms of.  We cache with a
        # dictionary, but numpy objects don't have hashes by default.
        outgrad_hash = int(hashlib.sha1(np.atleast_1d(outgrad).view(np.uint8)).hexdigest(), 16)
        if not self._grad.has_key((other,outgrad_hash)):
            self._grad[(other,outgrad_hash)] = self.compute_grad(other, outgrad)
        return self._grad[(other,outgrad_hash)]

    def compute_value(self, reset, rng, inputs):
        raise Exception("Class 'Differentiable' is abstract.")

    def compute_grad(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def local_grad(self, outgrad):
        raise Exception("Class 'Differentiable' is abstract.")

    def depends(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def shape(self, inputs=None):
        raise Exception("Class 'Differentiable' is abstract.")

from constants      import Constant, Parameter
from batcher        import Batcher
from inputs         import Inputs
from targets        import Targets
from matrix_ops     import MatAdd, MatMult, MatSum, Transpose, Reshape
from elem_ops       import ElemAdd
from nonlinearities import SoftReLU, HardReLU, LogSoftMax
from losses         import L2Loss, LogMultinomialLoss
from dropout        import Dropout
from regularizers   import L2Norm, L1Norm, Horseshoe, NExp
from crossval       import CrossValidator
