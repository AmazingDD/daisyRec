cimport numpy as np  # noqa
import numpy as np
from collections import defaultdict

from six.moves import range
from six import iteritems

def jaccard(n_x, yr, xr, min_support):
    cdef np.ndarray[np.int_t, ndim=2] freq
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj, yi
    cdef double ri, rj
    cdef int min_sprt = min_support

    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)

    u_is = defaultdict(list)
    for x, x_ratings in iteritems(xr):
        for yi, ri in x_ratings:
            u_is[x].append(yi)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = len(set(u_is[xi]) | set(u_is[xj]))
                sim[xi, xj] = len(set(u_is[xi]) & set(u_is[xj])) / denum

            sim[xj, xi] = sim[xi, xj]

    return sim

def cosine(n_x, yr, xr, min_support):
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim

def pearson(n_x, yr, xr, min_support):
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # sum (rxy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (rx'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # sum (rxy) for common ys
    cdef np.ndarray[np.double_t, ndim=2] si
    # sum (rx'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    freq = np.zeros((n_x, n_x), np.int)
    prods = np.zeros((n_x, n_x), np.double)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    si = np.zeros((n_x, n_x), np.double)
    sj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                prods[xi, xj] += ri * rj
                freq[xi, xj] += 1
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2
                si[xi, xj] += ri
                sj[xi, xj] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):

            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                                (n * sqj[xi, xj] - sj[xi, xj]**2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    return sim
