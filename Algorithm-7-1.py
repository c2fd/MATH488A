import numpy as np
from numpy.linalg import norm

def classical_gram_schmidt(A):
    # classical gram schmidt algorithm (unstable)
    m, n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    for j in range(n):
        vj = A[:,j]
        for i in range(j-1):
            R[i,j] = np.dot(Q[:,i], A[:,j])
            vj = vj - R[i,j]*Q[:,i]
        R[j,j] = norm(vj, 2)
        Q[:,j] = 1/R[j,j]*vj
    return Q, R

A = np.array([[1, 0],[0, 1],[1, 0]])
print('A', A)
Q, R = classical_gram_schmidt(A)
print(Q,R)
