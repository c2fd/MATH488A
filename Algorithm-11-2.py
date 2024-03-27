import numpy as np

def least_squares_QR(A, b):
    Q, R = np.linalg.qr(A)
    Qtb = Q.T@b
    x = np.linalg.solve(R, Qtb)
    return x

A = np.array([[1, -6], [1, -2], [1, 1], [1, 7]])
b = np.array([[-1],[2],[1],[6]])
x = least_squares_QR(A, b)
print(x)
