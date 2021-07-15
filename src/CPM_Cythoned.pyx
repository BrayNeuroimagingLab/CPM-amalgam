import Cython
from scipy.special import btdtr
import numpy as np
cimport numpy as np


@Cython.boundscheck(False)
@Cython.wraparound(False)
def get_Xtr(double[:,:] FC_flat, double[:] target, double low, double high, bint pselect):

    cdef:
        int p = FC_flat.shape[1]
        int n = FC_flat.shape[0]
        double sum_X = 0, sum_Y = 0, sum_XY = 0, squareSum_X = 0, squareSum_Y = 0

        int i1

    for i1 in range(n):
        sum_Y += target[i1]
        squareSum_Y += target[i1] * target[i1]

    cdef:
        int i
        int j
        double corr
        int dim = 0
    inds = []
    ab = n / 2 - 1
    for j in range(p):
        i = 0
        sum_X = 0
        squareSum_X = 0
        sum_XY = 0

        for i in range(n):
            sum_X += FC_flat[i][j]
            squareSum_X += FC_flat[i][j] * FC_flat[i][j]
            sum_XY += FC_flat[i][j] * target[i]

        corr = (n * sum_XY - sum_X * sum_Y) \
               / np.sqrt((n * squareSum_X - sum_X * sum_X)
                      * (n * squareSum_Y - sum_Y * sum_Y))

        if not pselect:
            if low <= corr and corr < high:
                inds.append(j)
                dim += 1
        elif pselect:
            prob = 2 * btdtr(ab, ab, 0.5 * (1 - abs(corr)))
            if low <= prob and prob < high:
                inds.append(j)
                dim += 1
    return inds, dim
