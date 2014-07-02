import numpy        as np
import numpy.random as npr

class Batcher(object):
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

    def __init__(self, batch_size, total_size, rng=None):
        """Constructor for the Kayak Batcher class.

        This creates the Batcher, which makes it easy to manage
        mini-batch indices for inputs and outputs.  This allows you to
        iterate through things in the order provided, or in a random
        order.

        Arguments:

          batch_size (int): Size of the mini-batches to produce.

          total_size (int): Total number of data to iterate over.

          rng (optional): Specifies whether the mini-batches should be
                          random or not.  If rng=None (default), then
                          the mini-batch indices will be in numeric
                          order, i.e., 0, 1, 2, 3, ...  If rng is an
                          integer, that will be used as the random
                          seed to produce a random permutation of the
                          indices per epoch.  The parameter rng can
                          also be a Numpy RandomState object, which
                          will be used instead of creating another one
                          with the specified seed.

        """
        if rng is None:
            self.rng = None
        elif type(rng) == int:
            self.rng = npr.RandomState(rng)
        else:
            self.rng = rng

        self.batch_size = batch_size
        self.total_size = total_size

        self.reset()

    def reset(self):
        """Reset the state of the Kayak Batcher.

        It may happen that you want to 'reset the loop' and restart
        your iteration over the data.  Calling this method does that.
        If, in the constructor, you set rng=None, then you'll go back
        to zero.  If rng was an integer or a RandomState object, you
        will get a new random permutation when you reset.

        This method is automatically called when the iterator
        completes its loop, so you don't need to explicitly call it
        when you're making multiple loops over the data.

        Arguments: None

        """

        if self.rng is None:
            self.ordering = np.arange(self.total_size, dtype=int)
        else:
            self.ordering = self.rng.permutation(self.total_size)
        self.start    = 0
        self.end      = min(self.start+self.batch_size, self.total_size)
        self._indices = self.ordering[self.start:self.end]

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

        self._indices = self.ordering[self.start:self.end]
        self.start += self.batch_size
        self.end    = min(self.start + self.batch_size, self.total_size)

        return self._indices

    def indices(self):
        """Get the current list of indices.

        You may want to get access to the mini-batch indices without
        advancing the iterator.  This method does that.  It just
        returns the indices in the current mini-batch.  This is how
        the Inputs and Targets classes access the indices.

        Arguments: None

        """

        return self._indices
