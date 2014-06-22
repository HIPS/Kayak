import numpy        as np
import numpy.random as npr

class Batcher(object):

    def __init__(self, batch_size, total_size, rng=None):
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
        if self.start >= self.total_size:
            self.reset()
            raise StopIteration

        self._indices = self.ordering[self.start:self.end]
        self.start += self.batch_size
        self.end    = min(self.start + self.batch_size, self.total_size)

        return self._indices

    def indices(self):
        return self._indices
