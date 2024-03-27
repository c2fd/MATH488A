import numpy as np

m, n = 100, 15
t = np.linspace(0,m-1,m)/(m-1)
t = t.reshape(-1,1)
print(t)

A = []
for i in range(n):
    if i == 0:
        A = t**i
    else:
        A = np.hstack((A, t**i))
    #print('A', A)
print('A',A.shape)
b = np.exp(np.sin(4*t))
b =  b/2006.787453080206
print(b[0])


x = np.linalg.lstsq(A, b)[0]
print('x',x)
y = A@x
kappa = np.linalg.cond(A,2)
print(kappa)
theta = np.arcsin(np.linalg.norm(b-y,2)/np.linalg.norm(b,2))
print(theta)
eta = np.linalg.norm(A,2)*np.linalg.norm(x, 2)/np.linalg.norm(y,2)
print(eta)
