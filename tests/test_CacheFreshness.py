import numpy        as np
import numpy.random as npr

import kayak

from . import *

# def test_MatSum_clears_cache():
#     X = kayak.Inputs(np.array([[1, 2, 3], [2, 3, 4]]))
#     Y = kayak.
                     
def test_batcher_clears_value_cache():
    batcher = kayak.Batcher(1, 2)
    X = kayak.Inputs(np.array([[1, 2, 3], [2, 3, 4]]), batcher)
    Y = kayak.MatSum(X)
    correct_vals = [6, 9]
    for ii, batch in enumerate(batcher):
        assert Y.value == correct_vals[ii]

def test_batcher_clears_shape_cache():
    batcher = kayak.Batcher(2, 3)
    X = kayak.Inputs(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), batcher)
    Y = kayak.MatSum(X, axis=1)
    correct_shapes = [(2, 1), (1, 1)]
    for ii, batch in enumerate(batcher):
        assert Y.shape == correct_shapes[ii]

def test_dropout_clears_value_cache():
    X = kayak.Inputs(np.random.randn(10,10))
    Y = kayak.Dropout(X)
    Z = kayak.MatSum(Y, axis=1)
    val1 = Z.value
    Y.draw_new_mask()
    val2 = Z.value
    assert not np.all(val1 == val2)
    assert np.all(Z.value == Z.value)

def test_data_update_clears_value_cache():
    X = kayak.Inputs(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))
    assert np.all(X.value == [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    X.data = [1,2]
    assert X._value is None
    assert np.all(X.value == [1, 2])

def test_param_change_clears_value_cache():
    pass

def test_param_change_clears_grad_cache():
    pass
