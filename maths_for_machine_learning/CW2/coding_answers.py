import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


B = np.array([[4, -2], [-2, 4]])
a = np.array([0, 1])
b = np.array([-2, 1])

x_test = np.array([-0.6, -0.4])

# f1(x) function
def f1(x):

    return np.matmul(x.T, np.matmul(B, x)) - np.matmul(x.T, x) + np.matmul(a.T, x) - np.matmul(b.T, x)


# f2(x) function
def f2(x):

    return np.cos(np.matmul((x-b).T, (x-b))) + np.matmul((x-a).T, np.matmul(B, (x-a)))


# f3(x) function
def f3(x):

    term1 = np.exp(-1 * np.matmul((x-a).T, (x-a)))
    term2 = np.exp(-1 * np.matmul((x-b).T, np.matmul(B, (x-b))))
    term3 = -(1/10) * np.log(np.linalg.det((1/100) * np.identity(2) + np.matmul(x, x.T)))

    return 1 - (term1 + term2 + term3)


# Gradient of f1(x)
def grad_f1(x):

    return (2 * np.matmul(x.T, B)) - (2 * x.T) + a.T - b.T


# Gradient of f2(x)
def grad_f2(x):

    return (-2 * np.sin(np.matmul((x-b).T, (x-b))) * (x - b).T) + (2 * np.matmul((x - a).T, B))


# Gradient of f3(x)
def grad_f3(x):

    term1 = -2 * np.exp(-1 * np.matmul((x-a).T, (x-a))) * (x-a).T

    term2 = -2 * np.exp(-1 * np.matmul((x-b).T, np.matmul(B, (x-b)))) * np.matmul((x-b).T, B)

    k = (1/100) * np.identity(2) + np.outer(x, x)

    x1 = x[0]  # unpack x1 from x
    x2 = x[1]  # unpack x2 from x

    # dxxT/dx is a 2x2x2 tensor
    dxxT = np.array([
        [[2 * x1, 0], [x2, x1]],
        [[x2, x1], [0, 2 * x2]]
    ])

    term3 = (-1 / 10) * np.trace(np.matmul(np.linalg.inv(k), dxxT))

    return -1 * (term1 + term2 + term3)


# gradient descent function
def gradient_decent(f, grad_f, pos, step_size, iteration=50):

    pos_list = []  # list of all positions from gradient descent
    step = []
    z = []  # list of all evaluation at position

    new_pos = pos

    pos_list.append(new_pos)  # add starting position to pos_list
    z.append(f(new_pos))  # evaluate the function at starting position

    for i in range(iteration):

        new_pos = new_pos - step_size * grad_f(new_pos)  # gradient descent

        pos_list.append(new_pos)  # add new position to list

        z.append(f(new_pos))  # evaluate function at new position

        step.append(i)

    final_pos = pos_list[-1]  # return final position after iterations

    return final_pos, pos_list, step, z

# Show the graphs
def graphs(f, grad_f, start_pos, step_size, num_iter=50, show_f=True, show_g=True):

    x1 = x2 = np.arange(-4, 4, 0.05)  # controls the range and spacing between evaluations of the graph

    X1, X2 = np.meshgrid(x1, x2)  # create mesh grid

    # evaluate function at all points
    fs = np.asarray([f(np.asarray([i1, i2])) for i1, i2 in zip(np.ravel(X1), np.ravel(X2))])

    F = fs.reshape(X1.shape)

    # evaluate gradient of the function at all points
    gs = np.asarray([np.linalg.norm(grad_f(np.asarray([i1, i2]))) for i1, i2 in zip(np.ravel(X1), np.ravel(X2))])

    G = gs.reshape(X1.shape)

    # gradient descent
    min_pos, pos, i, z = gradient_decent(f, grad_f, np.asarray(start_pos), step_size=step_size, iteration=num_iter)

    # figure set up
    fig = plt.figure(num=None, figsize=(20, 7), facecolor='w', edgecolor='r')

    # ax1: the 3d wire frame of the function
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_wireframe(X1, X2, F, color='grey', alpha=0.5)

    ax1.set_xlabel('$x1$')
    ax1.set_ylabel('$x2$')
    ax1.set_zlabel('$f(x)$')

    # ax2: contours of the function / gradient of the function
    ax2 = fig.add_subplot(1, 2, 2)

    # contour of the function
    if show_f:
        height = ax2.contourf(X1, X2, F, 50, linewidths=1)
        fig.colorbar(height)
    # contour of the gradient
    if show_g:
        gradients = ax2.contour(X1, X2, G, 20, cmap='jet', linewidths=1)
        fig.colorbar(gradients)

    # plot the gradient descent path as a red line on both plots
    for ind in range(len(pos) - 1):
        ax1.plot([pos[ind][0], pos[ind + 1][0]], [pos[ind][1], pos[ind + 1][1]], [z[ind], z[ind + 1]], 'r', linewidth=1)

    for ind in range(len(pos) - 1):
        ax2.plot([pos[ind][0], pos[ind + 1][0]], [pos[ind][1], pos[ind + 1][1]], 'r', linewidth=1)

    print('Final position: {}'.format(min_pos))
    print('Gradient at final position: {}'.format(grad_f(min_pos)))

    plt.show()

################################### READ ME ###################################

# graphs(f, grad_f, start_pos, step_size, num_iter=50, show_f=True, show_g=True):
#
# f, choice of function: f1, f2, f3
# grad_f, choice of gradient(must correlate with f): grad_f1, grad_f2, grad_f3
# start_pos, starting position: [x1, x2]
# step_size: float
# num_iter: number of steps to do gradient descent
# show_f, show contour of function: boolean
# show_g, show contour of gradient: boolean

graphs(f3, grad_f3, [0.3, 0], 0.27, num_iter=50)
