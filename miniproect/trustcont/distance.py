import math
import numpy

def d_euclidean(a,b):
    '''Returns euclidean distance between vectors a and b'''
    return math.sqrt(sum([(x - y)**2 for (x,y) in zip(a, b)]))

def distance_matrix(x, distf = None):
    """Returns distance matrix of pairwise distances between vectors
    stored in array a
    distf is function used to calculate distance between vectors,
    if None, euclidean distance is used
    """

    if distf == None:
        distf = d_euclidean

    vectors = x.shape[0]
    d = numpy.zeros([vectors, vectors])

    for i in range(vectors):
        for j in range(i, vectors):
            dd = distf(x[i, :], x[j, :])
            d[i, j] = dd
            d[j, i] = dd

    return d


def rank_matrix(x):
    """Returns rank matrix from pairwise distance matrix a"""

    m = x.argsort()
    r = numpy.zeros(x.shape)

    vectors = x.shape[0]

    for i in range(vectors):
        for j in range(vectors):
            pos = numpy.where(m[i, :] == j)
            r[i, j] = pos[0][0]  # there should be a better syntax for this

    return r.astype('int')


def centering(x):
    """Center matrix x to origo"""
    return x - x.mean(axis=0)


def double_centering(x):
    """Double center matrix x"""
    pass




