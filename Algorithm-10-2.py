import numpy as np
import math
sign = lambda x: math.copysign(1, x)
from numpy.linalg import norm as norm


# def fast_HouseHolderQR_ls_v2(A, b):
#     m, n = A.shape
#     b = b.reshape(-1,1)
#     R = np.array(A)
#     Qtb = np.array(b)
#
#     for k in range(n):
#         x = A[k:m, k]
#         e1 = 0 * x
#         e1[0] = 1
#         vk = sign(x[0]) * norm(x,2) * e1 + x
#         vk = vk / norm(vk,2)
#         vk = vk.reshape((-1,1))
#         A[k:m,k:n] = A[k:m,k:n] - 2 * vk @ (vk.T@A[k:m,k:n])
#         b[k:m,:] = b[k:m,:] - 2 * vk @ (vk.T @ b[k:m,:])
#     print('b.shape', b.shape)
#     print('a.shape,', A.shape)
#     x_sol = np.linalg.solve(A[0:n,0:n], b[0:n])
#     return x_sol

def fast_HouseHolderQR_ls(A, b):
    m, n = A.shape
    b = b.reshape(-1,1)
    for k in range(n):
        x = A[k:m, k]
        e1 = 0 * x
        e1[0] = 1
        vk = sign(x[0]) * norm(x,2) * e1 + x
        vk = vk / norm(vk,2)
        vk = vk.reshape((-1,1))
        A[k:m,k:n] = A[k:m,k:n] - 2 * vk @ (vk.T@A[k:m,k:n])
        b[k:m,:] = b[k:m,:] - 2 * vk @ (vk.T @ b[k:m,:])
    print('b.shape', b.shape)
    print('a.shape,', A.shape)
    x_sol = np.linalg.solve(A[0:n,0:n], b[0:n])
    return x_sol

A = np.array([[1.,-6.],[1.,-2.],[1.,1.],[1.,7.]])
b = np.array([-1.,2.,1.,6.])
x = fast_HouseHolderQR_ls(A,b)
print(x)