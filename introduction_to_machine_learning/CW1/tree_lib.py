import numpy as np
import matplotlib.pyplot as plt

# DECISION TREE IMPLEMENTATION

class Node:
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left, self.right = left, right
    
    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def display(self, max_depth=5, offset_x=0.5, offset_y=1, n=1):
        if np.log2(n) < max_depth:
            plt.text(offset_x, 
                     offset_y, 
                     "X{} < {}".format(self.attribute, self.value), 
                     size=10, 
                     ha="center", 
                     va="center", 
                     bbox=dict(facecolor='white')
                    )
            plt.plot(np.linspace(offset_x, offset_x+2/n, 2), np.linspace(offset_y, offset_y-0.4, 2))
            plt.plot(np.linspace(offset_x, offset_x-2/n, 2), np.linspace(offset_y, offset_y-0.4, 2))

            self.left.display(max_depth, offset_x-2/n, offset_y-0.4, n*2)
            self.right.display(max_depth, offset_x+2/n, offset_y-0.4, n*2)

    def predict(self, mesure):
        if mesure[self.attribute] < self.value:
            return self.left.predict(mesure)
        else:
            return self.right.predict(mesure)
    
    def list_of_prunable(self, path=""):
        if type(self.left) is Leaf and type(self.right) is Leaf:
            return [path]
        else:
            return self.left.list_of_prunable(path+"l") + self.right.list_of_prunable(path+"r")
    
    def pruned(self, path =""):
        if path == "":
            return self.left, self.right
        elif path[0] == "r":
            first, second = self.right.pruned(path[1:])
            return Node(self.attribute, self.value, self.left, first), Node(self.attribute, self.value, self.left, second)
        else:
            first, second = self.left.pruned(path[1:])
            return Node(self.attribute, self.value, first, self.right), Node(self.attribute, self.value, second, self.right) 
  

class Leaf:
    def __init__(self, label):
        self.label = label
    
    def depth(self):
        return 1    

    def predict(self, mesure=None):
        return self.label
    
    def list_of_prunable(self, path=None):
        return []
    
    def display(self, max_depth=5, offset_x=0.5, offset_y=1, n=None):
        if np.log2(n) < max_depth:
            plt.text(offset_x, 
                     offset_y, 
                     str(self.label),
                     size=10,
                     ha="center",
                     va="center", 
                     bbox=dict(facecolor='white')
                    )
    
    

# TRAINING FUNCTION AND RELATED PROCEDURES

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
    best_feature = None
    best_split = None
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
                    best_split = (sort_dataset[j][i] + sort_dataset[j+1][i])/2
                    best_feature = i
                    split_index = j

    return best_feature, best_split, split_index

def decision_tree_learning(training_dataset, depth=1):
    labels = np.unique(training_dataset[:, -1])
    if len(labels) == 1:
        return Leaf(labels[0]), depth
    else:
        feature, split, idx = find_split(training_dataset)
        dataset = sort_feature(training_dataset, feature)

        l_branch, l_depth = decision_tree_learning(dataset[:idx+1], depth+1)
        r_branch, r_depth = decision_tree_learning(dataset[idx+1:], depth+1)

        return Node(feature, split, l_branch, r_branch), max(l_depth, r_depth)


# FUNCTIONS TO EVALUATE A MODEL

def predict(model, test):
        labels = []
        preds = []
        for i in test:
            labels.append(i[-1])
            preds.append(model.predict(i[:-1]))
        return np.array(labels), np.array(preds)

def class_rate(model, test):    
        labels, preds = predict(model, test)
        return np.mean(np.array(labels) == np.array(preds))

def confusion_matrix(model, test):
        labels, preds = predict(model, test)
        M = np.array([[0 for i in range(4)] for j in range(4)])
        for i in range(len(labels)):
            M[int(labels[i])-1][int(preds[i])-1] += 1
        return np.array(M)

def recall(confusion):
        L = []
        for i in range(4):
            tp = confusion[i][i]
            f = sum([confusion[i][j] for j in range(4)])
            L.append(tp/f)
        return np.array(L)

def precision(confusion):
        L = []
        for i in range(4):
            tp = confusion[i][i]
            f = sum([confusion[j][i] for j in range(4)])
            L.append(tp/f)
        return np.array(L)

def F1(confusion):
        p = precision(confusion)
        r = recall(confusion)
        return 2*p*r/(p + r)

def evaluate(model, test):
    M = confusion_matrix(model, test)
    c = class_rate(model, test)
    p = precision(M)
    r = recall(M)
    F = F1(M)
    return M, c, p, r ,F

# DISPLAYING CONFUSION MATRIX IN A CONVENIENT WAY

def show_confusion_matrix(confusion):

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, cmap='GnBu')
    fig.colorbar(cax)

    alpha = ['Room 1', 'Room 2', 'Room 3', 'Room 4']

    ax.set_xticklabels(['']+alpha)
    ax.set_yticklabels(['']+alpha)

    plt.xlabel('Prediction')
    ax.xaxis.set_label_position('top')
    plt.ylabel('True Label')

    for (i, j), z in np.ndenumerate(confusion):
        ax.text(j, i, '{}'.format(int(z)), ha='center', va='bottom')

    plt.show()


# PRUNING

def pruning(model, test):
    To_inspect = model.list_of_prunable()
    
    while (len(To_inspect) != 0): 
        Parents = []
        for x in To_inspect:
            model_left, model_right = model.pruned(x)
            
            score1 = class_rate(model, test)
            score2 = class_rate(model_left, test)
            score3 = class_rate(model_right, test)
            best = max(score1,score2,score3)
            
            if score2 == best:
                model = model_left
                Parents.append(x[:-1])
            if score3 == best:
                model = model_right
                Parents.append(x[:-1])
        
        To_inspect = [value for value in model.list_of_prunable() if value in Parents]
    
    return model


# CROSS VALIDATION

def cross_validation(dataset, k=10, prune=False, show_matrix=False):
    np.random.shuffle(dataset)
    L = np.split(dataset, k)
    M, c, p, r , F = [], [], [], [], []
    Depth = []
    for i in range(k):
        test = L[i]
        if prune:            
            A = L[:i] + L[i+1:]
            for j in range(k-1):
                validation = A[j]
                train = np.concatenate(A[:j] + A[j+1:])
                model, d = decision_tree_learning(train, 0)
                model = pruning(model, validation)
                e1, e2, e3, e4, e5 = evaluate(model, test)
                M.append(e1)
                c.append(e2)
                p.append(e3)
                r.append(e4)
                F.append(e5)
                Depth.append(model.depth())    
        else:
            train = np.concatenate(L[:i] + L[i+1:])
            model, d = decision_tree_learning(train, 1)
            e1, e2, e3, e4, e5 = evaluate(model, test)
            M.append(e1)
            c.append(e2)
            p.append(e3)
            r.append(e4)
            F.append(e5)
            Depth.append(d)
        print("fold {} done".format(i))        
    M, c, p, r , F = np.array(M).mean(axis=0), np.array(c).mean(axis=0), np.array(p).mean(axis=0), np.array(r).mean(axis=0), np.array(F).mean(axis=0)
    Depth = np.array(Depth)
    if show_matrix:
        show_confusion_matrix(M)
    return M, c, p, r , F, Depth.mean(), Depth.std()


# THE CLASS LEARNER

# This class doesn't add any functionnality but provides a nice interface to manipulate trees and run tests efficiently

class Learner:
    def __init__(self, train, test, validation=None):
        self.train, self.test = train, test
        self.model = None
        self.confusion = None
        self.validation = validation
        
        print("Train and test set loaded")
        if validation is None:
            print("Warning: no validation set was provided, the test set will be used for pruning")
    
    def learn(self, verbose=False):
        m, a = decision_tree_learning(self.train, 1)
        self.model = m
        M = self.confusion_matrix()
        if verbose:
            print("Tree generated, the maximum depth is {}".format(a))
    
    def predict(self, data_to_predict=None):
        if data_to_predict is None:
            return predict(self.model, self.test)
        else:
            return predict(self.model, data_to_predict)
    
    def class_rate(self):
        return class_rate(self.model, self.test)
    
    def confusion_matrix(self):
        M = confusion_matrix(self.model, self.test)
        self.confusion = M
        return M
    
    def recall(self):
        return recall(self.confusion)
    
    def precision(self):
        return precision(self.confusion)
    
    def F1(self):
        return F1(self.confusion)
    
    def display_tree(self, max_depth=5):
        plt.figure(figsize=(20, 10), dpi=100)
        plt.axis('off')
        self.model.display(max_depth)
    
    def pruning(self):
        if self.validation is None:
            self.model = pruning(self.model, self.test)
        else: 
            self.model = pruning(self.model, self.validation)
        M = self.confusion_matrix()
    
    def show_confusion_matrix(self):
        show_confusion_matrix(self.confusion)
    
    def evaluate(self):
        return evaluate(self.model, self.test)
    
    def model_depth(self):
        return self.model.depth()