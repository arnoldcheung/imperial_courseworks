import numpy as np
import matplotlib.pyplot as plt
import math

beta = math.gamma(24)/(math.gamma(8)*math.gamma(16))

print(beta)

mu = 0.3

prob = beta * (mu**7)*((1-mu)**15)

print(prob)

x = np.linspace(0, 1, 100)



plt.figure(1)
plt.plot(x, [beta * (mu**7)*((1-mu)**15) for mu in x])
plt.show()