# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

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

    def __init__(self, batch_size, total_size, random_batches=False):
        """Constructor for the Kayak Batcher class.

        This creates the Batcher, which makes it easy to manage
        mini-batch indices for inputs and outputs.  This allows you to
        iterate through things in the order provided, or in a random
        order.

        Arguments:

          batch_size: (Integer) Size of the mini-batches to produce.

          total_size: (Integer) Total number of data to iterate over.

          random_batches: (Bool) Specifies whether the mini-batches
                          should be random or not.
        """
        super(Batcher, self).__init__([])
        self.batch_size = batch_size
        self.total_size = total_size
        self.random_batches = random_batches
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
        if self.random_batches:
            self.ordering = npr.permutation(self.total_size)
        else:
            self.ordering = np.arange(self.total_size, dtype=int)
        self.start    = 0
        self.end      = min(self.start+self.batch_size, self.total_size)
        self._value = self.ordering[self.start:self.end]

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
        if self.start >= self.total_size:
            self.reset()
            raise StopIteration

        self._clear_value_cache()

        self._value = self.ordering[self.start:self.end]
        self.start += self.batch_size
        self.end    = min(self.start + self.batch_size, self.total_size)

        return self._value

