import numpy as np
import copy

edge_prob = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.1, 0.1, 0.0],
                      [0.0, 0.1, 1.0, 0.1, 0.0],
                      [0.0, 0.1, 0.1, 0.1, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]])

compatibilities = {('e', 'e'): 2, ('e', 'ne'): 1, ('ne', 'e'): 1, ('ne', 'ne'): 1}

labels = ['e', 'ne']


def C(i, j, x, y):
    if ((x == i + 1 or x == i - 1) and (y == j)) or ((x == i) and (y == j + 1 or y == j - 1)):
        return 1
    else:
        return 0


def prob(prob_matrix, i, j, label):
    if label == 'e':
        return prob_matrix[i, j]
    elif label == 'ne':
        return 1 - prob_matrix[i, j]


def relaxation_labelling(labels, prob_mat, compatibilities, iteration):

    new_prob_mat = copy.copy(prob_mat)

    for s in range(iteration):

        for i in range(prob_mat.shape[0]):
            for j in range(prob_mat.shape[1]):



                Cqs_e = []
                Cqs_ne = []

                for x in range(prob_mat.shape[0]):
                    for y in range(prob_mat.shape[1]):

                        # print((i, j))
                        # print((x, y))
                        # print(C(i,j,x,y))

                        Cqs_e.append(C(i, j, x, y) * sum([compatibilities[('e', k)] * prob(prob_mat, x, y, k) for k in labels]))
                        Cqs_ne.append(C(i, j, x, y) * sum([compatibilities[('ne', k)] * prob(prob_mat, x, y, k) for k in labels]))


                Q_e = sum(Cqs_e)
                Q_ne = sum(Cqs_ne)

                new_prob_mat[i, j] = (prob(prob_mat, i, j, 'e') * Q_e) / ((prob(prob_mat, i, j, 'e') * Q_e) + (prob(prob_mat, i, j, 'ne') * Q_ne))

    return new_prob_mat

new_prob = relaxation_labelling(labels, edge_prob, compatibilities, 1)

print(relaxation_labelling(labels, new_prob, compatibilities, 1))
