import numpy as np

def Cholesky_decomposition(A):
    #input: A (SPD)
    #output: R^TR = A
    m, n = A.shape
    assert np.allclose(A, A.T)
    eigvals = np.linalg.eigvals(A)
    assert np.all(eigvals > 0)

    R = np.array(A)
    for k in range(m):
        for j in range(k+1,m):
            R[j,j:m] = R[j,j:m] - R[k,j:m] * R[k,j]/R[k,k]
        R[k,k:m] = R[k,k:m]/(R[k,k])**0.5
    for i in range(m):
        for j in range(0,i):
            R[i,j] = 0.
    return R

A = np.array([[4,2,6],[2,2,5],[6,5,22]], dtype=np.float64)
R = Cholesky_decomposition(A)
print(R.T)





