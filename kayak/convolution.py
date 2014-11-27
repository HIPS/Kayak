# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np

import util

from . import Differentiable
import sys

class Convolve1d(Differentiable):
    __slots__ = ['A','B','ncolors']
    def __init__(self, A, B, ncolors=1):
        super(Convolve1d, self).__init__([A,B])
        self.A    = A
        self.B    = B
        self.ncolors = ncolors

    def _compute_value(self):
        A = self.A.value
        B = self.B.value
        filtersize = B.shape[0]/self.ncolors

        # Broadcast to get color channels
        A = np.reshape(A, (A.shape[0], self.ncolors, -1))
        D = A.shape[-1] - filtersize + 1
        output = np.zeros((A.shape[0], D, B.shape[1]))

        offset = 0
        for j in xrange(D):
            output[:,j,:] = np.dot(A[:, :,offset:offset+filtersize].reshape((A.shape[0],-1)), B)
            offset += 1

        return output.reshape((A.shape[0], D*B.shape[1]))

    def _local_grad(self, parent, d_out_d_self):
        filtersize = self.B.shape[0]/self.ncolors
        if parent == 0:
            output     = np.zeros((self.A.shape[0], self.A.shape[1]))
            B          = self.B.value
            output = output.reshape((output.shape[0], self.ncolors, -1))
            outgrad = d_out_d_self.reshape(d_out_d_self.shape[0], -1, B.shape[-1])
            for i in xrange(outgrad.shape[1]):
                output[:,:,i:i+filtersize] += np.dot(outgrad[:,i,:], B.T).reshape((output.shape[0], self.ncolors, filtersize)) 

            return output.reshape((output.shape[0], -1))

        elif parent == 1:
            output     = np.zeros((self.B.shape[0], self.B.shape[1]))
            A          = self.A.value
            A          = np.reshape(A, (A.shape[0], self.ncolors, -1))
            filtersize = self.B.shape[0]/self.ncolors
            outgrad    = np.reshape(d_out_d_self, (d_out_d_self.shape[0], -1, self.B.shape[1]))
            offset = 0
            for j in xrange(outgrad.shape[1]):
                output += np.dot(A[:,:,offset:offset+filtersize].reshape((A.shape[0],-1)).T, outgrad[:,j,:])
                offset += 1
            return output
        else:   
            raise Exception("Not a parent of me")

class Pool(Differentiable):
    __slots__ = ['A', 'width', 'indices', 'ncolors']

    def __init__(self, A, width, ncolors=1):
        super(Pool, self).__init__([A])
        self.A        = A
        self.width    = width
        self.ncolors  = ncolors
        self.indices  = None

    def _compute_value(self):
        A = self.A.value
        try:
            A = np.reshape(A, (A.shape[0], self.ncolors, -1, self.width))
        except:
            print A.shape
            print (A.shape[0], self.ncolors, -1, self.width)
            raise
        self.indices = np.argmax(A, axis=3)
        x, z, t = np.indices(self.indices.shape)
        A = A[x, z, t, self.indices]
        A = A.reshape((self.A.shape[0],-1))
        return A

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            mask = np.zeros(self.A.shape).reshape((self.A.shape[0], self.ncolors, -1, self.width))
            inds, inds2, inds3 = np.indices(self.indices.shape)
            mask[inds, inds2, inds3, self.indices] = d_out_d_self.reshape((mask[inds, inds2, inds3, self.indices].shape))
            mask = mask.reshape((self.A.shape[0], -1))
            return mask
        else:
            raise Exception("Not a parent of me")

class TopKPool(Differentiable):
    __slots__ = ['A', 'k', 'indices', 'ncolors']

    def __init__(self, A, k, ncolors=1):
        super(TopKPool, self).__init__([A])
        self.A        = A
        self.k        = k
        self.ncolors  = ncolors
        self.indices  = None

    def _compute_value(self):
        A = self.A.value.copy()
        A = np.reshape(A, (A.shape[0], self.ncolors, -1))
        self.indices = np.argsort(A, axis=2)[:,:,-self.k:]
        a, b, c = np.indices(self.indices.shape)
        A = A[a, b, self.indices]
        return A.reshape((self.A.shape[0],-1))

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            mask = np.zeros(self.A.shape).reshape((self.A.shape[0], self.ncolors, -1))
            inds, inds2, inds3 = np.indices(self.indices.shape)
            mask[inds, inds2, self.indices] = d_out_d_self.reshape((mask[inds, inds2, self.indices].shape))
            mask = mask.reshape((self.A.shape[0], -1))
            return mask
        else:
            raise Exception("Not a parent of me")
# class MaxPool(Differentiable):
#     __slots__ = ['X', 'width']

#     def __init__(self, X, width, axis=1):
#         super(Take, self).__init__([X])

#         self.X         = X
#         self._inds     = inds
#         self._axis     = axis

#     def _compute_value(self):
#         slice_list = [slice(None), ] * self.X.value.ndim
#         slice_list[self._axis] = self._inds
#         return self.X.value[slice_list]

#     def _local_grad(self, parent, d_out_d_self):
#         result = np.zeros(self.X.shape)
#         slice_list = [slice(None), ] * result.ndim
#         slice_list[self._axis] = self._inds
#         result[slice_list] = d_out_d_self
#         return result

