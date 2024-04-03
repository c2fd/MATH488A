import numpy as np

def gaussian_elimination_with_partial_pivoting(A):
    m, n = A.shape
    assert (m == n)
    U = np.array(A)
    L = np.identity(m)
    P = np.identity(m)

    for k in range(m-1):
        i = np.argmax(np.abs(U[k:,k])) + k
        tmp_U = np.array(U[k,k:m])
        U[k,k:m] = U[i,k:m]
        U[i,k:m] = tmp_U[:]

        tmp_L = np.array(L[k,0:k])
        L[k,0:k] = L[i,0:k]
        L[i,0:k] = tmp_L

        tmp_P = np.array(P[k,:])
        P[k,:] = P[i,:]
        P[i,:] = tmp_P

        for j in range(k+1,m):
            ljk = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - ljk * U[k,k:m]
            L[j,k] = ljk

    return L, U, P

def solve_linear_system_guassian_elimination(A, b):
    import scipy
    L, U, P = gaussian_elimination_with_partial_pivoting(A) #PA=LU
    # Ax = b ---> PA= Pb ---> LUx=Pb
    w = scipy.linalg.solve(L, P.dot(b))  #Lw = Pb
    x = scipy.linalg.solve(U, w)  #Ux = w

    return x

A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]], dtype=np.float64)

L, U, P = gaussian_elimination_with_partial_pivoting(A)
print('L', L)
print('U', U)
print('P', P)

b = np.array([[4],[11],[29],[30]], dtype=np.float64)
x = solve_linear_system_guassian_elimination(A, b)
print(x)



