#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt', dtype=int)
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

class Node:
    def __init__(self, attribute, value, left=None, right=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
    def predict(self, mesure):
        if mesure[self.attribute] < self.value:
            return self.left.predict(mesure)
        else:
            return self.right.predict(mesure)

class Leaf:
    def __init__(self, label):
        self.label = label
    def predict(self, mesure):
        return self.label


def entropy(dataset):
    labels, labels_count = np.unique(dataset[:, -1], return_counts=True)
    tot = len(dataset)
    return -sum([c/tot*np.log2(c/tot) for c in labels_count])

def gain(data_all, data_left, data_right):
    tot = len(data_all)
    return entropy(data_all) - (entropy(data_left)*len(data_left)/tot + entropy(data_right)*len(data_right)/tot)

def sort_feature(dataset, feature):
    return np.array(sorted(dataset, key=lambda x: x[feature]))

def find_split(dataset):
    max_feature = None
    max_split = None
    max_gain = -1
    split_index = None

    n_features = len(dataset[0]) - 1
    n_data = len(dataset) - 1
    for i in range(n_features):
        sort_dataset = sort_feature(dataset, i)
        for j in range(n_data):
            if sort_dataset[j][i] != sort_dataset[j+1][i]:
                g = gain(sort_dataset, sort_dataset[:j+1], sort_dataset[j+1:])
                if g > max_gain:
                    max_gain = g
                    max_split = (sort_dataset[j][i] + sort_dataset[j+1][i])/2
                    max_feature = i
                    split_index = j
    return max_feature, max_split, split_index

def decision_tree_learning(training_dataset, depth):
    labels = np.unique(training_dataset[:, -1])
    if len(labels) == 1:
        return Leaf(labels[0]), depth
    else:
        feature, split, idx = find_split(training_dataset)
        dataset = sort_feature(training_dataset, feature)
        l_branch, l_depth = decision_tree_learning(dataset[:idx+1], depth+1)
        r_branch, r_depth = decision_tree_learning(dataset[idx+1:], depth+1)
        return Node(feature, split, l_branch, r_branch), max(l_depth, r_depth)

def predict(test_dataset, model):
    labels = []
    preds = []
    for i in test_dataset:
        labels.append(i[-1])
        preds.append(model.predict(i[:-1]))
    return labels, preds

def accuracy(test_dataset, model):    
    labels, preds = predict(test_dataset, model)
    return np.mean(np.array(labels) == np.array(preds))

def evaluate(dataset):
    L = np.split(dataset, 10)
    S = []
    for i in range(10):
        train = np.concatenate(L[:i] + L[i+1:])
        test = L[i]
        model, d = decision_tree_learning(train, 0)
        S.append(accuracy(test, model))
    return np.array(S).mean()


def confusion_matrix(labels,preds):
    M = np.array([[0 for i in range(4)] for j in range(4)])
    for i in range(len(labels)):
        M[int(labels[i])-1][int(preds[i])-1] += 1
    return M


# In[ ]:





# In[2]:


idx = 1500


# In[3]:


np.random.shuffle(clean_dataset)
train = clean_dataset[:idx]
test = clean_dataset[idx:]
model, d = decision_tree_learning(train, 0)
accuracy(test, model)


# In[101]:


np.random.shuffle(noisy_dataset)
train = noisy_dataset[:idx]
test = noisy_dataset[idx:]
model, d = decision_tree_learning(train, 0)
accuracy(test, model)


# In[6]:




if type(model) is Node:
    print(model.left)
    print(model.right)


# In[ ]:





# In[61]:


labels, preds = predict(test, model)
confusion_matrix(labels,preds)


# In[67]:


[0,1] + [2,3]


# In[ ]:


import numpy as np

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt', dtype=int)
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

class Node:
    def __init__(self, attribute, value, left=None, right=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
    def display(self, indent=""):
        print(indent + "Node({}, {})".format(self.attribute, self.value))
        self.left.display(indent + "| ")
        self.right.display(indent + "| ")
    def evaluate(self, mesure):
        if mesure[self.attribute] < self.value:
            return self.left.evaluate(mesure)
        else:
            return self.right.evaluate(mesure)

class Leaf:
    def __init__(self, label):
        self.label = label
    def display(self, indent=""):
        print(indent + "Leaf({})".format(self.label))
    def evaluate(self, mesure):
        return self.label

