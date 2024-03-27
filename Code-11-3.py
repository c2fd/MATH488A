import numpy as np
def least_squares_SVD(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(S)
    Utb = U.T@b
    w = np.linalg.solve(Sigma, Utb)
    x = Vt.T@w
    return x


A = np.array([[1, -6], [1, -2], [1, 1], [1, 7]])
b = np.array([[-1],[2],[1],[6]])
x = least_squares_SVD(A, b)
print(x)