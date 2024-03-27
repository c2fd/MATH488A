import numpy as np
from numpy.linalg import norm

def modified_gram_schmidt(A):
    m, n = A.shape
    V = np.zeros((m,n))
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    for i in range(n):
        V[:,i] = A[:,i]

    for i in range(n):
        R[i,i] = norm(V[:,i], 2)
        Q[:,i] = 1/R[i,i] * V[:,i]
        for j in range(i+1,n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:,j] = V[:,j] - R[i,j] * Q[:,i]
    return Q, R

A = np.array([[1, 0],[0, 1],[1, 0]])
print('A', A)
Q, R = modified_gram_schmidt(A)
print(Q,R)
