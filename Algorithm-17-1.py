import numpy as np

def back_substitution(R, b):
    m, n = R.shape
    assert (m == n)
    x = np.zeros(b.shape)
    for j in range(m-1,-1,-1):
        x[j] = b[j]
        for k in range(j+1,m):
            x[j] = x[j] - x[k] * R[j,k]
        x[j] = x[j]/R[j,j]
    return x

R = np.array([[3,1,2],[0,2,1],[0,0,1]], dtype=np.float64)
b = np.array([[6],[3],[1]])
#Rx = b
x = back_substitution(R, b)
print('solution for Rx=b', x)


