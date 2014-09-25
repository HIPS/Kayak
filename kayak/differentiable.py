import numpy as np

class Differentiable(object):

    def __init__(self, parents=[]):
        self._value = None
        self._grad  = {}
        self._shape = {}
        for parent_index, parent in enumerate(parents):
            if isinstance(parent, Differentiable):
                parent._add_child(self, parent_index)

        self._parents = parents
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
        # Check that the old (pre 2014-09-25) isn't being used
        if isinstance(rng, bool):
            raise Exception("New (9/2014) interface does not accept 'reset' argument.")

        # # No need to recurse at all if this object's value is already determined.
        if inputs is not None and inputs.has_key(self):
            return inputs[self]

        # If we're resetting, or if the value is not yet cached, compute it.
        if self._value is None:
            self._value = self._compute_value(rng, inputs)

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
        return other._d_out_d_self(self)

    def shape(self, inputs=None):
        if inputs in self._shape:
            return self._shape[inputs]

        shape = self._compute_shape(inputs)
        self._shape[inputs] = shape
        return shape

    def _d_out_d_self(self, out):
        if out in self._grad:
            return self._grad[out]

        if self is out:
            grad = 1.0
        elif len(self._children) == 0:
            grad = np.zeros(self.shape())
        else:
            grad = sum([child._d_out_d_parent(out, parent_index) for 
                        child, parent_index in self._children])

        self._grad[out] = grad
        return grad

    def _d_out_d_parent(self, out, parent):
        return self._local_grad(parent, self._d_out_d_self(out))

    def _clear_value(self):
        """Recursively clears cached value and gradient by maintaining
        the invariant that if a node's _value is None that node's
        descendants' values and gradients are also None (or {}). This
        is the logical dual of saying that if a node does have a
        value, so must its parents."""

        if self._value is None:
            # Node is already clear
            return

        self._clear_grad()

        for child, _ in self._children:
            child._clear_value()

        self._value = None

    def _clear_grad(self):
        """Recursively clears gradient and those gradients that
        explicitly depend on it via backprop. If a node's _grad is
        empty, the node's parents' _grad variables are also empty."""
        if not self._grad:
            return

        for parent in self._parents:
            # TODO: could check here whether you actually need to look at
            # all the parents. (e.g. if C = A + B, changing A doesn't change
            # dC/dB)
            if isinstance(parent, Differentiable):
                parent._clear_grad()

        self._grad = {}

    def _local_grad(self, parent, d_out_d_self):
        """Return d_out_d_self * d_self_d_parent"""
        raise Exception("Class 'Differentiable' is abstract.")

    def _add_child(self, child, parent_index):
        """We need to keep track of our children. parent_index tells us which
        parent we are."""
        self._children.append((child, parent_index))

    def _compute_value(self, rng, inputs):
        raise Exception("Class 'Differentiable' is abstract.")

    def _compute_shape(self, rng, inputs):
        raise Exception("Class 'Differentiable' is abstract.")

