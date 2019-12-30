import numpy as np

D = [[1, 2, 3], [-1, 0, 0], [-4, 4, 2]]

x_bar = np.mean(D, axis=0)

cov = np.cov(D, bias=True)

DT = np.asarray(D).T

covT = np.cov(DT, bias=True)

print(DT)

print(covT)

def q_ik(i, k, data_set, N):

    return (1/N)*sum([(data_set[i][j]-np.mean(data_set[i]))*(data_set[k][j]-np.mean(data_set[k])) for j in range(N)])


print(x_bar)

#x_bar = 3*x_bar

a = np.asarray((D[0]-x_bar)).reshape(3, 1)
cov1 = np.matmul(a, a.T)

b = np.asarray((D[1]-x_bar)).reshape(3, 1)
cov2 = np.matmul(b, b.T)

c = np.asarray((D[2]-x_bar)).reshape(3, 1)
cov3 = np.matmul(c, c.T)

cov_t = (cov1 + cov2 + cov3)/3

print(cov_t*9)

print(cov1)
print(cov2)
print(cov3)