import numpy as np
import matplotlib.pyplot as plt

xy = [[0, 5.5], [1, 6], [2.5, 6.5], [3.5, 7], [4.5, 7.5], [6, 8], [7.5, 8.5], [9, 9], [11.9, 9.5], [14.3, 10]]

# ms = np.linspace(0, 1, 10)
#
# fig, ax = plt.subplots(figsize=(7, 6))
#
# for pos in xy:
#     plt.plot(ms, [(pos[1] - m * pos[0]) for m in ms], label=pos, linewidth=0.5)
#
#
# major_ticks_m = np.arange(0, 1.1, 0.1)
# minor_ticks_m = np.arange(0, 1.1, 0.02)
# major_ticks_c = np.arange(0, 11, 1)
# minor_ticks_c = np.arange(0, 11, 0.2)
#
# ax.set_xticks(major_ticks_m)
# ax.set_xticks(minor_ticks_m, minor=True)
# ax.set_yticks(major_ticks_c)
# ax.set_yticks(minor_ticks_c, minor=True)
#
# plt.ylim(0, 10)
# plt.xlim(0,1)
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
# plt.xlabel('m')
# plt.ylabel('c')
# ax.legend()

x = [i[0] for i in xy]
y = [i[1] for i in xy]

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')

x_est = np.linspace(0, 15, 30)
y_est_1 = x_est * 0.38 + 5.85

y_est_2 = x_est * 0.4 + 5.6
y_est_3 = x_est * 0.32 + 6.0

plt.plot(x_est, y_est_1, c='r', label='m = 0.38, c = 5.85')
plt.plot(x_est, y_est_2, c='lime', label='m = 0.4, c = 5.6')
plt.plot(x_est, y_est_3, c='g', label='m = 0.32, c = 6.0')

plt.legend()
plt.show()


# y = mx + c