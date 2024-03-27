import numpy as np
from numpy.linalg import norm
import math

sign = lambda x: math.copysign(1, x)

def HouseHolder_QR(A_ori):
    m, n = A_ori.shape
    A = np.array(A_ori)

    for k in range(n):
        x = A[k:m,k]
        e1 = 0*x
        e1[0] = 1
        vk = sign(x[0]) * norm(x,2) * e1 + x
        vk = vk / norm(vk, 2)
        vk = np.reshape(vk, [-1,1])
        A[k:m,k:n] = A[k:m,k:n] - 2. * vk @ (vk.T @ A [k:m,k:n])
    return A  # R

def HouseHolder_QR_withQ(A_ori):
    A = np.array(A_ori)
    m, n = A.shape
    Q = np.eye(m, dtype=float)
    print("Q",Q)
    for k in range(n):
        x = A[k:m,k]
        e1 = 0*x
        e1[0] = 1
        vk = sign(x[0]) * norm(x,2) * e1 + x
        vk = vk / norm(vk, 2)
        vk = np.reshape(vk, [-1,1])
        A[k:m,k:n] = A[k:m,k:n] - 2. * vk @ (vk.T @ A [k:m,k:n])
        H = np.eye(m)
        H[k:m,k:m] = np.eye(vk.shape[0], dtype=float) - 2. * vk @ vk.T
        Q = Q@H
    R = A
    return Q, R

# def HouseHolder_QR(A, b):
#     m, n = A.shape
#     for k in range(n):
#         x = A[k:m,k]
#         #x = x.reshape(-1,1)
#         print('x',x)
#         e1 = 0*x
#         e1[0] = 1
#         vk = sign(x[0]) * norm(x,2) * e1 + x
#         print('norm 2', norm(x,2))
#         vk = vk / norm(vk, 2)
#         A[k:m,k:n] = A[k:m,k:n] - 2 * np.outer(vk,vk) @ A [k:m,k:n]
#         b[k:m] = b[k:m] - 2 * np.outer(vk,vk) @ b[k:m]
#     R = A
#     return R


#A = np.array([[1, 0],[0, 1],[1, 0]])
A = np.array([[2., -2.,  18.],[2., 1., 0.],[1., 2., 0.]])
print('A', A)
R = HouseHolder_QR(A)
print(R)

Q, R = HouseHolder_QR_withQ(A)
print('R', R)
print('Q', Q)
print('QR-', Q@R -A)