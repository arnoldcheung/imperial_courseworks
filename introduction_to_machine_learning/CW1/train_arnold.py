import sys
from operator import add

import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, feature, value, entropy_gain, depth, l_len, r_len, leaf=False, decision=None):
        self.left_node = None
        self.right_node = None
        self.feature = feature
        self.value = value
        self.entropy_gain = entropy_gain
        self.depth = depth
        self.l_len = l_len
        self.r_len = r_len
        self.leaf = leaf
        self.decision = decision

    def __str__(self):
        if self.leaf:
            return 'Depth: {}   Leaf: {}   # of data: {}'.format(self.depth, int(self.decision), self.l_len)
        else:
            return 'Depth: {}   Feature {} >= {}?   L: {} R: {}'.format(self.depth, self.feature, self.value, self.l_len, self.r_len)

    def get_left(self):
        return self.left_node

    def get_right(self):
        return self.right_node


def sort_feature(array, feature):

    return np.asarray(sorted(array, key=lambda data: data[feature]))


def get_entropy(array, label_col):
    unique, unique_count = np.unique(array[:, label_col], return_counts=True)
    count_dict = dict(zip(unique, unique_count))

    return -1 * sum([(count_dict[key]/len(array))*np.log2((count_dict[key]/len(array))) for key in unique])


def get_entropy_gain(original_array, array_1, array_2):

    return get_entropy(original_array, 7) - ((len(array_1)/len(original_array))*get_entropy(array_1, 7) + (len(array_2)/len(original_array))*get_entropy(array_2, 7))


def split_data(array, depth=0, print_info=True):

    entropy_gains = []
    max_entropy_gain = 0
    num_split = 0

    best_feature = None
    best_split_value_1 = None
    best_split_value_2 = None

    best_split_1 = []
    best_split_2 = []

    for feature in range(np.size(array, axis=1) - 1):

        array = sort_feature(array, feature)
        feature_num_split = 0

        for i in range(len(array)-1):
            if array[i, feature] == array[i+1, feature]:
                continue

            else:
                num_split += 1
                feature_num_split += 1

                split_1 = array[0:i+1]
                split_2 = array[i+1:]

                entropy_gain = get_entropy_gain(array, split_1, split_2)
                entropy_gains.append(entropy_gain)

                if entropy_gain > max_entropy_gain:
                    best_feature = feature

                    best_split_value_1 = split_1[-1, feature]
                    best_split_value_2 = split_2[0, feature]

                    max_entropy_gain = entropy_gain

                    best_split_1 = split_1
                    best_split_2 = split_2

    if max_entropy_gain == 0:
        best_split_1 = array[0:1]
        best_split_2 = array[1:]

    if print_info:
        print('''
        Depth: {}
        Split feature: {}
        Split between: {} and {}
        Split data 1 length: {}
        Split data 2 length: {}
        Entropy gain: {}
        '''.format(depth, best_feature, best_split_value_1, best_split_value_2, len(best_split_1), len(best_split_2), max_entropy_gain))

    return [best_split_1, best_split_2], best_feature, best_split_value_1, len(best_split_1), len(best_split_2), max_entropy_gain


def decision_tree_learning(training_data_set, depth, nodes):

    if get_entropy(training_data_set, 7) == 0:

        leaf_node = Node(None, None, 0, depth, len(training_data_set), 0, leaf=True, decision=np.unique(training_data_set[:, np.size(training_data_set, axis=1)-1]))
        nodes.append(leaf_node)
        return leaf_node, depth

    else:

        data, feature, value, l_len, r_len, entropy_gain = split_data(training_data_set, depth, print_info=False)

        branch_node = Node(feature, value, entropy_gain, depth, l_len, r_len)
        branch_node.left_node, l_depth = decision_tree_learning(data[0], depth+1, nodes)
        branch_node.right_node, r_depth = decision_tree_learning(data[1], depth+1, nodes)
        nodes.append(branch_node)
        return branch_node, max(l_depth, r_depth)


def train(training_data_set):

    nodes = []
    decision_tree_learning(training_data_set, 0, nodes)
    return nodes






