import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_indices_1():
    """Test with deterministic indices."""

    for num_data in [1, 10, 100, 1000, 10000, 100000]:
        for batch_size in [1, 10, 11, 25, 50, 101, 500, 1000, 1011]:
            
            data_used = np.zeros((num_data,), dtype=bool)
            batcher = kayak.Batcher(batch_size, num_data)
            for batch in batcher:
                data_used[batch] = True
            
            assert np.all(data_used)

def test_indices_2():
    """Test with random seed."""
    npr.seed(1)

    for num_data in [1, 10, 100, 1000, 10000, 100000]:
        for batch_size in [1, 10, 11, 25, 50, 101, 500, 1000, 1011]:
            
            data_used = np.zeros((num_data,), dtype=bool)
            batcher = kayak.Batcher(batch_size, num_data, random_batches=True)
            for batch in batcher:
                data_used[batch] = True
            
            assert np.all(data_used)

def test_reset():
    """Test resetting."""

    for num_data in [1000, 10000, 100000]:
        for batch_size in [1, 10, 11, 25, 50, 101, 500]:
            
            batcher = kayak.Batcher(batch_size, num_data)

            # Start the batcher forward.
            batcher.next()

            # Now reset.
            batcher.reset()

            # Make sure we touch all of the data.
            data_used = np.zeros((num_data,), dtype=bool)
            for batch in batcher:
                data_used[batch] = True
            
            assert np.all(data_used)

def test_batcher_updates_value():
    batcher = kayak.Batcher(12, 20)
    data = npr.randn(20, 7)
    X = kayak.Inputs(data, batcher)
    for i, batch in enumerate(batcher):
        if i == 0:
            assert np.all(X.value == data[:12, :])
        elif i == 1:
            assert np.all(X.value == data[12:, :])
        else:
            assert False
    
    batcher.test_mode()
    assert np.all(X.value == data)

def test_batcher_updates_dropout():
    batcher = kayak.Batcher(5, 10)
    X = kayak.Inputs(np.random.randn(10,10))
    Y = kayak.Dropout(X, batcher=batcher)
    val1 = Y.value
    batcher.next()
    val2 = Y.value
    assert not np.all(val1 == val2)

def test_batcher_can_reinstate_dropout_mask():
    batcher = kayak.Batcher(5, 10)
    X = kayak.Inputs(np.ones((10,10)))
    Y = kayak.Dropout(X, batcher=batcher)
    assert not np.all(Y.value == np.ones((10, 10)))
    batcher.test_mode()
    print "Y value", Y.value
    assert np.all(Y.value == np.ones((10, 10)))
