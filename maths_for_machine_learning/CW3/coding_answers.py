import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

################## Question 1 ##################

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (-1, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)


K = [1, 11]


def polynomial_phi(X, K):

    X = X.flatten()

    return np.array([X ** j for j in range(K + 1)]).T


def trigonometry_phi(X, K):

    X = X.flatten()

    N = X.shape[0]

    Phi = np.zeros((N, 2 * K + 1)).T

    Phi[0] = 1

    for j in range(1, K+1):
        Phi[2 * j - 1] = np.sin(2 * np.pi * j * X)
        Phi[2 * j] = np.cos(2 * np.pi * j * X)

    return Phi.T


def w_mle(Phi, Y):

    kappa = 1e-8

    D = Phi.shape[1]

    return np.linalg.inv(Phi.T.dot(Phi) + kappa * np.identity(D)).dot(Phi.T).dot(Y)


def plot(basis, K_list):

    test_x = np.reshape(np.linspace(-1, 1.2, 200), (-1, 1))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)

    for K in K_list:
        Phi = basis(X, K)
        w = w_mle(Phi, Y)

        test_Phi = basis(test_x, K)
        pred_y = test_Phi @ w

        ax.plot(test_x, pred_y, label='K = {}'.format(K))

    plt.ylim((-3, 3))
    plt.xlabel('x')
    plt.ylabel('y')
    ax.legend()
    plt.show()


def cross_validation(basis, max_K, folds, X, Y):

    X = X.flatten()
    Y = Y.flatten()

    combined = list(zip(X, Y))
    random.shuffle(combined)

    shuffled_X, shuffled_Y = zip(*combined)

    shuffled_X = np.reshape(shuffled_X, (-1, 1))
    shuffled_Y = np.reshape(shuffled_Y, (-1, 1))

    data_X = np.split(shuffled_X, folds)
    data_Y = np.split(shuffled_Y, folds)

    K_list = []
    average_MSE_list = []
    sd_mle_list = []

    for K in range(max_K + 1):

        average_MSE = 0
        K_list.append(K)

        for i in range(folds):
            train_set_X = np.concatenate(data_X[:i] + data_X[i + 1:])
            train_set_Y = np.concatenate(data_Y[:i] + data_Y[i + 1:])

            test_set_X = data_X[i]
            test_set_Y = data_Y[i]

            Phi = basis(train_set_X, K)
            w = w_mle(Phi, train_set_Y)

            test_Phi = basis(test_set_X, K)
            pred_Y = test_Phi @ w

            MSE = (sum((test_set_Y - pred_Y) ** 2)) / len(pred_Y)
            average_MSE += MSE


        Phi = basis(X, K)
        w = w_mle(Phi, Y)

        test_Phi = basis(X, K)
        pred_Y = test_Phi @ w
        sd_mle = sum((Y - pred_Y) ** 2) / len(X)
        sd_mle_list.append(sd_mle)

        average_MSE = average_MSE / folds
        average_MSE_list.append(average_MSE)

    fig, ax = plt.subplots()
    ax.plot(K_list, average_MSE_list, label='Average squared error')
    ax.plot(K_list, sd_mle_list, label='Maximum likelihood value of $σ^2$')
    plt.xlabel('K')
    plt.legend()

    plt.show()


plot(polynomial_phi, [0, 1, 2, 3, 11])  # Q1a

plot(trigonometry_phi, [1, 11])  # Q1b

cross_validation(trigonometry_phi, 10, 25, X, Y)  # Q1c


################## Question 2 ##################

# Q2a
def lml(alpha, beta, Phi, Y):

    D = Phi.shape[1]
    N = len(Y)

    mean = 0
    cov = Phi @ (alpha * np.identity(D)) @ Phi.T + (beta * np.identity(N))
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    return (-1/2) * Y.T.dot(inv_cov).dot(Y) - (1/2) * np.log(det_cov) - (N/2) * np.log(2 * np.pi)


# Q2a
def grad_lml(alpha, beta, Phi, Y):

    D = Phi.shape[1]
    N = len(Y)

    cov = alpha * (Phi @ Phi.T) + (beta * np.identity(N))
    inv_cov = np.linalg.inv(cov)

    da = (-1 / 2) * (np.trace(inv_cov @ Phi @ Phi.T) - Y.T @ (inv_cov @ Phi @ Phi.T @ inv_cov) @ Y)
    db = (-1 / 2) * (np.trace(inv_cov) - Y.T @ (inv_cov @ inv_cov) @ Y)

    return np.ravel(da), np.ravel(db)


# Q2b
def gradient_descent(start_pos, iteration, step_size, f, grad_f, Phi, Y):
    pos_list = []
    lml = []

    current_alpha = start_pos[0]
    current_beta = start_pos[1]

    pos_list.append((current_alpha, current_beta))
    lml.append(f(current_alpha, current_beta, Phi, Y))

    for i in range(iteration):

        current_alpha = current_alpha + step_size * grad_f(current_alpha, current_beta, Phi, Y)[0][0]
        current_beta = current_beta + step_size * grad_f(current_alpha, current_beta, Phi, Y)[1][0]

        pos_list.append((current_alpha, current_beta))
        lml.append(f(current_alpha, current_beta, Phi, Y))

    return pos_list, lml


# Q2c
def question2c():
    k_list = []
    lml_list = []
    for k in range(13):
        Phi = trigonometry_phi(X, k)
        p, l = gradient_descent([0.25, 0.25], 20000, 0.00005, lml, grad_lml, Phi, Y)

        max_lml = max(l)
        print(max_lml)

        k_list.append(k)
        lml_list.append(max_lml[0][0])

    plt.plot(k_list, lml_list)
    plt.xlabel('Order of Basis Functions, K')
    plt.ylabel('Maximum Log Marginal Likelihood')

    plt.show()


# Q2d
def gaussian_phi(X, min_mean, max_mean, scale, space):

    X = X.flatten()
    mean = np.linspace(min_mean, max_mean, space)
    mean = mean.flatten()

    return np.array([np.exp(-((X - uj)**2)/(2*scale**2)) for uj in mean]).T


# Q2d
def parameter_posterior(alpha, beta, Phi, Y):

    D = Phi.shape[1]
    N = len(Y)

    SN = np.linalg.inv(np.linalg.inv(alpha*np.identity(D)) + (1/beta) * Phi.T.dot(Phi))
    mN = SN @ ((1/beta) * Phi.T.dot(Y))

    return mN, SN


# Q2d
def sample_posterior(mean, cov, num_samples):

    mean = mean.flatten()

    samples = np.random.multivariate_normal(mean, cov, num_samples)

    return samples

# Q2d
def plot_predict():

    Phi = gaussian_phi(X, -0.5, 1, 0.1, 10)

    mean, cov = parameter_posterior(1, 0.1, Phi, Y)

    samples = sample_posterior(mean, cov, 5)

    test_x = np.reshape(np.linspace(-1, 1.5, 200), (-1, 1))

    test_Phi = gaussian_phi(test_x, -0.5, 1, 0.1, 10)


    i = 0
    for sample in samples:
        i += 1
        y_pred = test_Phi @ sample
        plt.plot(test_x, y_pred, label='Sample {}'.format(i), linewidth=0.5)

    plt.scatter(X, Y, label='Real Data')

    std =np.sqrt(np.diagonal(cov)).reshape(Phi.shape[1], 1)

    mean_plus_two_std = mean + 2 * std
    mean_minus_two_std = mean - 2 * std

    y_mean = test_Phi @ mean

    y_plus_two_std = test_Phi @ mean_plus_two_std
    y_minus_two_std = test_Phi @ mean_minus_two_std

    plt.plot(test_x, y_mean, label='Mean', c='r')
    plt.plot(test_x, y_plus_two_std, label='$\pm2\sigma$', c='r', linestyle='--')
    plt.plot(test_x, y_minus_two_std, c='r', linestyle='--')


    plt.fill_between(test_x.flatten(), y_plus_two_std.flatten(), y_minus_two_std.flatten(), color='k', alpha=0.1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.show()


# Q2b
def question2b():
    Phi = polynomial_phi(X, 1)

    p, l = gradient_descent([0.5, 0.5], 100, 0.005, lml, grad_lml, Phi, Y)

    combined = (zip(p, l))

    for i, j in combined:
        print(i, j)

    alpha_range = np.arange(0.3, 0.6, 0.01)
    beta_range = np.arange(0.4, 0.5, 0.01)

    A, B = np.meshgrid(alpha_range, beta_range)

    fs = np.asarray([lml(i1, i2, Phi, Y) for i1, i2 in zip(np.ravel(A), np.ravel(B))])

    F = fs.reshape(A.shape)

    fig = plt.figure(num=None, figsize=(8, 7), facecolor='w', edgecolor='r')

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_wireframe(A, B, F, alpha=0.3, color='k', linewidth=1)

    ax.set_xlabel('$alpha$')
    ax.set_ylabel('$beta$')

    for ind in range(len(p) - 1):

        ax.plot([p[ind][0], p[ind + 1][0]], [p[ind][1], p[ind + 1][1]], zs=[l[ind][0][0], l[ind + 1][0][0]], c='r', linewidth=1)

    plt.show()


question2b()  # Q2b

question2c()  # Q2c

plot_predict()  # Q2d



################################################ NOTES ################################################
# **********************************************
# k = 1, a = 0.425, b = 0.45, lml = -27.6
# alpha_range = np.arange(0.3, 0.6, 0.01)
# beta_range = np.arange(0.4, 0.5, 0.01)
#
# Phi = polynomial_phi(X, 1)
#
#
# p, l = gradient_descent([0.48, 0.48], 100, 0.025, lml, grad_lml, Phi, Y)
#
# combined = (zip(p, l))
#
# for i, j in combined:
#     print(i, j)
# **********************************************


# **********************************************
# Phi = trigonometry_phi(X, 1)
#
#
# p, l = gradient_descent([0.25, 0.17], 100, 0.005, lml, grad_lml, Phi, Y)
#
# combined = (zip(p, l))
#
# for i, j in combined:
#     print(i, j)

# alpha_range = np.arange(0.15, 0.4, 0.005)
# beta_range = np.arange(0.12, 0.25, 0.005)
# **********************************************




# alpha_range = np.arange(0.4, 0.5, 0.005)
# beta_range = np.arange(0.42, 0.48, 0.005)


# k = 2, a = 15, b = 0.3, lml = -27.4
# alpha_range = np.arange(8, 50, 0.5)
# beta_range = np.arange(0.2, 0.5, 0.05)

# k = 3, a = 10, b = 0.26, lml = -26
# alpha_range = np.arange(8, 50, 0.5)
# beta_range = np.arange(0.2, 0.5, 0.05)

