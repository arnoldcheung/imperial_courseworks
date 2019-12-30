import numpy as np
from decimal import *

# getcontext().prec = 10
#
# D = 10
#
# I = 5
#
# r = 0.1
#
# sum = 0
# for t in range(10000):
#     term = Decimal((D + t * I)/((1 + r)**t))
#
#     sum = sum + term
#
# print(sum)
#
# print(D*(1 + r)/r + I*(1+r)/r**2)

s = [0.04, 0.06, 0.08]

P = 10/(1+0.04) + 110/(1+0.06)**2
C = 10
F = 100

print(P)

a = P
b = 2*P - C
c = (P - F - 2 * C)

Y = np.roots([a, b, c])[1]

D = (1 + Y)/Y - (1 + Y + 2 * (C/F - Y))/(((C/F) * ((1 + Y)**2 - 1)) + Y)

#
# f12 = ((1 + s[1])**2/(1 + s[0])) - 1
#
# print(f12)

def forward_rate(t1, t2):
    return (((1 + s[t2 - 1])**t2)/((1 + s[t1 - 1])**t1))**(1/(t2 - t1)) - 1

f12 = forward_rate(1, 2)
f13 = forward_rate(1, 3)

print(f12)
print(f13)

print(10/(1+f12) + 110/(1+f13)**2)



