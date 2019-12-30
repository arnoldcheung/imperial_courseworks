import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

class node:
    def __init__(self, feature, value, entropy_gain, depth, leaf=False, decision=None):
        self.left_node = None
        self._right_node = None
        self.feature = feature
        self.value = value
        self.entropy_gain = entropy_gain
        self.depth = depth
        self.leaf = leaf
        self.decision = decision

    def __str__(self):
        if self.leaf:
            return 'Depth: {}   Leaf: {}'.format(self.depth, self.decision)
        else:
            return 'Depth: {}   Feature {}>={}'.format(self.depth, self.feature, self.value)

    def get_left(self):
        return self.left_node

    def get_right(self):
        return self._right_node


def decision_tree_learning(training_dataset, depth):

    if


    else



    return(node, )

def FIND_SPLIT()


def sort()

def entropy(dataset):

    feature_col=dataset.shape[1]-1
    count=np.unique(dataset[:,feature_col], return_counts=True)

    return()



def gain(dataset, subleft,subright):

