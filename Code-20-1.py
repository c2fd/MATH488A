import  numpy as np

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

def guassian_elimination_without_pivoting(A):
    m, n = A.shape
    assert (m == n)
    U = np.array(A)
    L = np.identity(m)

    for k in range(m-1):
        for j in range(k+1, m):
            ljk = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - ljk * U[k,k:m]
            L[j,k] = ljk
    return L, U

def solve_linear_system_guassian_elimination(A, b):
    import scipy
    L, U = guassian_elimination_without_pivoting(A)
    w = scipy.linalg.solve(L, b)  #Lw = b
    #x = scipy.linalg.solve(U, w)  #Ux = w
    x = back_substitution(U, w)
    return x


A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]], dtype=np.float64)

L, U = guassian_elimination_without_pivoting(A)
print('L', L)
print('U', U)

b = np.array([[4],[11],[29],[30]], dtype=np.float64)
x = solve_linear_system_guassian_elimination(A, b)
print(x)