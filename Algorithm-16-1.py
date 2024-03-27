import numpy as np

def linear_algebra_QR(A, b):
    Q, R = np.linalg.qr(A,mode='reduced')
    y = Q.T @ b
    x = np.linalg.solve(R, y)
    return x


