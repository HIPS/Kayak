import numpy        as np
import numpy.random as npr
import itertools    as it

from constants import Parameter

def checkgrad(input, output, epsilon=1e-4):
    if not isinstance(input, Parameter):
        raise Exception("Cannot evaluate gradient in terms of non-Parameter type %s", (type(input)))

    # Need to make sure all evals have the same random number generation.
    rng_seed = 1

    value   = output.value(True, rng=npr.RandomState(rng_seed))
    an_grad = output.grad(input)
    fd_grad = np.zeros(an_grad.shape)

    for in_dims in it.product(*map(range, input.shape())):
        input.value()[in_dims] += epsilon/2.0
        fn_up = output.value(True, rng=npr.RandomState(rng_seed))
        input.value()[in_dims] -= epsilon
        fn_dn = output.value(True, rng=npr.RandomState(rng_seed))
        input.value()[in_dims] += epsilon/2.0
            
        fd_grad[in_dims] = (fn_up - fn_dn)/epsilon

        #print in_dims, (an_grad[in_dims] - fd_grad[in_dims])/np.abs(fd_grad[in_dims]), an_grad[in_dims], fd_grad[in_dims]
        
    return np.mean(np.abs((an_grad - fd_grad)/fd_grad))
            
def logsumexp(X, axis=None):
    maxes = np.max(X, axis=axis)
    return np.log(np.sum(np.exp(X - maxes), axis=axis)) + maxes
