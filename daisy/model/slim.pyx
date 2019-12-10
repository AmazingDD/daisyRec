import numpy
cimport numpy
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

cdef inline double max_double(double x, double y):
    return x if x >= y else y

cdef inline double min_double(double x, double y):
    return x if x <= y else y

cdef inline double abs_double(double x):
    return x if x >= 0 else -x

cdef inline double soft_thresholding(double a, double b):
    if b >= abs_double(a):
        return 0
    if a > 0:
        return a - b
    if a < 0:
        return 0


def compute_covariance(numpy.ndarray[numpy.double_t, ndim=2] A, int start, int end):
    cdef int n = A.shape[1]
    cdef numpy.ndarray[numpy.double_t, ndim=2] covariance_array = numpy.empty((end-start, n), numpy.double)
    cdef int row, j
    for row in range(end-start):
        for j in range(row+start, n):
            covariance_array[row, j] = A[:, row+start].dot(A[:, j])
    return covariance_array


def symmetrize_covariance(numpy.ndarray[numpy.double_t, ndim=2] covariance_array):
    cdef int n = covariance_array.shape[1]
    cdef int i, j
    for i in range(1, n):
        for j in range(i):
            covariance_array[i, j] = covariance_array[j, i]


def coordinate_descent(double alpha, double lam_bda, int max_iter, double tol, double N, int p, numpy.ndarray[numpy.double_t, ndim=2] covariance_array, int start, int end):
    cdef numpy.ndarray[numpy.double_t] gradient_components = numpy.empty(p, numpy.double)
    cdef double b = lam_bda * alpha * N
    cdef double c = lam_bda * (1 - alpha) * N

    cdef numpy.ndarray[numpy.double_t, ndim=2] W = numpy.zeros((p, end-start), numpy.double)
    cdef int step, col, j, k, move, mode
    cdef double a, new_Wj, delta
    for col in range(end-start):
        gradient_components[:] = 0
        mode = 0
        for step in range(max_iter):
            move = 0
            for j in range(p):
                if j == col+start:
                    continue

                if mode == 1 and W[j, col] == 0:
                    continue

                a = covariance_array[j, col+start] + covariance_array[j, j] * W[j, col] - gradient_components[j]
                new_Wj = soft_thresholding(a, b) / (c + covariance_array[j, j])

                delta = new_Wj - W[j, col]
                if abs_double(delta) > tol:
                    W[j, col] = new_Wj
                    move = 1
                    for k in range(p):
                        gradient_components[k] += covariance_array[k, j] * delta
            if move == 0:
                if mode == 0:
                    break
                if mode == 1:
                    mode = 0
            elif mode == 0:
                mode = 1
    return W


def coordinate_descent_lambda_ratio(double alpha, double ratio, int max_iter, double tol, double N, int p, numpy.ndarray[numpy.double_t, ndim=2] covariance_array, int start, int end):
    cdef numpy.ndarray[numpy.double_t] gradient_components = numpy.empty(p, numpy.double)
    cdef numpy.ndarray[numpy.double_t, ndim=2] W = numpy.zeros((p, end-start), numpy.double)
    cdef int step, col, j, k, move, mode
    cdef double a, new_Wj, delta, max_cov, b, c
    for col in range(end-start):
        gradient_components[:] = 0

        max_cov = 0
        for j in range(p):
            if j != col+start:
                max_cov = max_double(max_cov, covariance_array[j, col+start])
        if max_cov == 0:
            continue
		
        b = max_cov * ratio  # lam_bda * alpha * N
        c = max_cov * (1 - alpha) / alpha * ratio  # lam_bda * (1 - alpha) * N
        mode = 0
        for step in range(max_iter):
            move = 0
            for j in range(p):
                if j == col+start:
                    continue

                if mode == 1 and W[j, col] == 0:
                    continue

                a = covariance_array[j, col+start] + covariance_array[j, j] * W[j, col] - gradient_components[j]
                new_Wj = soft_thresholding(a, b) / (c + covariance_array[j, j])
                
                delta = new_Wj - W[j, col]
                if abs_double(delta) > tol:
                    W[j, col] = new_Wj
                    move = 1
                    for k in range(p):
                        gradient_components[k] += covariance_array[k, j] * delta
            if move == 0:
                if mode == 0:
                    break
                if mode == 1:
                    mode = 0
            elif mode == 0:
                mode = 1

    return W