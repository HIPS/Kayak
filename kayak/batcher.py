# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy        as np
import numpy.random as npr

from . import Differentiable

class Batcher(Differentiable):
    """Kayak class for managing batches of data.
    
    This class is intended to provide a simple interface for managing
    mini-batches of data, both on the input side and on the output
    side.  It can be set up to either use random minibatches, or go
    through the data in the order provided.  You tell it how many data
    you have and how large the mini-batches should be.  It will
    provide a sequence of indices via an iterator for easy looping.

    To use this class, you would do something like this:

    # Create an instance of the batcher.
    kyk_batcher = Batcher( batch_size, num_data )

    # When you create input and output objects, give them access to
    # the batcher.
    kyk_inputs  = Inputs(X, kyk_batcher)
    kyk_targets = Targets(Y, kyk_batcher)

    # Probably you'll loop over training epochs.
    for epoch in xrange(num_epochs):

      # Then you can treat the batcher as an iterator.
      for batch in kyk_batcher:

        # Do your mini-batch training here.

    """
    __slots__ = ['_rng', '_batch_size', '_total_size', '_random_batches',
                 '_dropout_nodes', 'start', 'end', 'ordering']
    def __init__(self, batch_size, total_size, random_batches=False, rng=None):
        """Constructor for the Kayak Batcher class.

        This creates the Batcher, which makes it easy to manage
        mini-batch indices for inputs and outputs.  This allows you to
        iterate through things in the order provided, or in a random
        order.

        Arguments:

          batch_size: (Integer) Size of the mini-batches to produce.

          total_size: (Integer) Total number of data to iterate over.

          _random_batches: (Bool) Specifies whether the mini-batches
                          should be random or not.
        """
        super(Batcher, self).__init__([])

        if rng is None:
            self._rng = npr.RandomState()
        else:
            self._rng = rng

        self._batch_size = batch_size
        self._total_size = total_size
        self._random_batches = random_batches
        self._dropout_nodes = []
        self.reset()

    def reset(self):
        """Reset the state of the Kayak Batcher.

        It may happen that you want to 'reset the loop' and restart
        your iteration over the data.  Calling this method does that.
        If, in the constructor, you set rng=None, then you'll go back
        to zero. If random_batches is true, you will get a new random
        permutation when you reset.

        This method is automatically called when the iterator
        completes its loop, so you don't need to explicitly call it
        when you're making multiple loops over the data.

        Arguments: None

        """
        self.start    = 0
        self.end      = min(self.start+self._batch_size, self._total_size)

        if self._random_batches:
            self.ordering = self._rng.permutation(self._total_size)
            self._value = self.ordering[self.start:self.end]
        else:
            self._value = slice(self.start, self.end)

        for node in self._dropout_nodes:
            node.draw_new_mask()

    def __iter__(self):
        return self

    def next(self):
        """Implementation of iterator functionality.

        The Batcher class is used as an iterator.  This method
        implements the iteration step forward.  It will return lists
        of indices that are the data in each mini-batch.  In general,
        these lists will be of size batch_size (as specified in the
        constructor).  The last one may be smaller, if the number of
        data is not an integer multiple of the batch size.

        Arguments: None

        """
        if self.start >= self._total_size:
            self.reset()
            raise StopIteration

        self._clear_value_cache()

        if self._random_batches:
            self._value = self.ordering[self.start:self.end]
        else:
            self._value = slice(self.start, self.end)

        self.start += self._batch_size
        self.end    = min(self.start + self._batch_size, self._total_size)

        for node in self._dropout_nodes:
            node.draw_new_mask()

        return self._value

    def add_dropout_node(self, node):
        self._dropout_nodes.append(node)

    def test_mode(self):
        """
        Turns off batching. Run before test-time.
        """
        self._clear_value_cache()
        self._value = slice(None, None) # All indices
        for node in self._dropout_nodes:
            node.reinstate_units()
