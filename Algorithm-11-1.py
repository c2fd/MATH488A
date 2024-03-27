import numpy as np

def least_squares_normal_equation(A, b):
    AtA = A.T@A
    Atb = A.T@b
    R = np.linalg.cholesky(AtA)
    w = np.linalg.solve(R.T, Atb)
    x = np.linalg.solve(R, w)
    return x


A = np.array([[1, -6], [1, -2], [1, 1], [1, 7]])
b = np.array([[-1],[2],[1],[6]])
x = least_squares_normal_equation(A, b)
print(x)
