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
    __slots__ = ['A', 'B', 'ncolors', 'stride']

    def __init__(self, A, B, ncolors=1, stride=1):
        super(Convolve1d, self).__init__([A,B])
        self.A       = A
        self.B       = B
        self.ncolors = ncolors
        self.stride  = stride

    def _compute_value(self):
        A = self.A.value
        B = self.B.value
        filtersize = B.shape[0]/self.ncolors

        # Broadcast to get color channels
        A = np.reshape(A, (A.shape[0], -1))

        D = A.shape[-1]/self.ncolors/self.stride - filtersize + 1
        output = np.zeros((A.shape[0], D, B.shape[1]))

        inds   = np.arange(filtersize)
        inds   = np.concatenate([inds+(i*A.shape[1]/self.ncolors) for i in xrange(self.ncolors)])
        for j in xrange(0, D):
            output[:,j,:] = np.dot(A[:, inds], B)
            inds   += self.stride

        return output.reshape((A.shape[0], D*B.shape[1]))

    def _local_grad(self, parent, d_out_d_self):
        A          = self.A.value
        A          = np.reshape(A, (A.shape[0], -1))        
        filtersize = self.B.shape[0]/self.ncolors
        inds   = np.arange(filtersize)
        inds   = np.concatenate([inds+(i*A.shape[1]/self.ncolors) for i in xrange(self.ncolors)])            

        if parent == 0:
            output     = np.zeros((self.A.shape))
            B          = self.B.value
            outgrad = d_out_d_self.reshape(d_out_d_self.shape[0], -1, B.shape[-1])

            for j in xrange(outgrad.shape[1]):
                output[:,inds] += np.dot(outgrad[:,j,:], B.T)
                inds += self.stride

            return output

        elif parent == 1:
            output     = np.zeros((self.B.shape[0], self.B.shape[1]))           
            outgrad    = np.reshape(d_out_d_self, (d_out_d_self.shape[0], -1, self.B.shape[1]))

            for j in xrange(0, outgrad.shape[1]):
                output += np.dot(A[:,inds].T, outgrad[:,j,:])
                inds   += self.stride

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

        # determine pooled shape variables
        conv_length = A.shape[1]/self.ncolors
        width_mod = conv_length % self.width
        width_aug = self.width - width_mod

        # augment convolution output to make pool width work
        if width_mod > 0:
            # insert at the back end of each convolution
            idx = np.ravel([[i*conv_length]*width_aug for i in range(1,self.ncolors+1)])
            
            # insert -inf
            A = np.insert(A, idx, -np.inf, axis=1)

        # bring together elements in a pooling group
        A = np.reshape(A, (A.shape[0], self.ncolors, -1, self.width))

        # get the index of the max within each pooling group
        self.indices = np.argmax(A, axis=3)

        # represent the first 3 dimensions of A
        x, z, t = np.indices(self.indices.shape)

        # index into the 4th dimension to pull out the maxes
        A = A[x, z, t, self.indices]

        # reshape back to the original form with the last dimension pooled
        A = A.reshape((self.A.shape[0],-1))
        
        return A

        '''
        try:
            A = np.reshape(A, (A.shape[0], self.ncolors, -1, self.width))
        except:
            print 'Could not pool with a width of %d on a layer of size %d' % (self.width, A.shape[0]/self.ncolors)
            print A.shape
            print (A.shape[0], self.ncolors, -1, self.width)
            raise
        '''

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            # determine pooled shape variables
            conv_length = self.A.shape[1]/self.ncolors
            width_mod = conv_length % self.width
            width_aug = self.width - width_mod
            pool_length = conv_length/self.width + 1*(width_mod>0)

            # create a zero matrix to match the reshaped version of A
            #  that brings together elements in a pool group
            mask = np.zeros((self.A.shape[0], self.ncolors, pool_length, self.width))

            # represent the first 3 dimensions of mask
            inds, inds2, inds3 = np.indices(self.indices.shape)

            # set the max indexes in mask to d_out_d_self,
            #  reshaped to fit the shape of this reduced version of the full matrix A
            mask[inds, inds2, inds3, self.indices] = d_out_d_self.reshape((mask[inds, inds2, inds3, self.indices].shape))

            # reshape to original form, with the last dimension pooled
            mask = mask.reshape((self.A.shape[0], -1))
            
            # remove the added dummy columns
            if width_mod > 0:
                conv_length_aug = conv_length + width_aug
                idx = [i*conv_length_aug-m for i in range(1,self.ncolors+1) for m in range(1,width_aug+1)]
                mask = np.delete(mask, idx, axis=1)

            return mask

            '''
            mask = np.zeros(self.A.shape).reshape((self.A.shape[0], self.ncolors, -1, self.width))
            inds, inds2, inds3 = np.indices(self.indices.shape)
            mask[inds, inds2, inds3, self.indices] = d_out_d_self.reshape((mask[inds, inds2, inds3, self.indices].shape))
            mask = mask.reshape((self.A.shape[0], -1))
            '''
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
