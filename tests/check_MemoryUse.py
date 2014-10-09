import numpy        as np
import numpy.random as npr
from guppy import hpy
import sys
sys.path.append('..')
import kayak

def check_NodeMemory():
    # Not a test. Useful for checking how much memory a node uses.
    np_A = npr.randn(5,6)
    A    = kayak.Parameter(np_A)
    N = int(1e4)
    h = hpy()
    h.setref()
    for i in xrange(N):
        A = kayak.Identity(A)
    print "Created 10,000 objects"
    print h.heap()

if __name__ == "__main__":
    check_NodeMemory()
