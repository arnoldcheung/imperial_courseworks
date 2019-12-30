from train_arnold import *
import copy

def print_tree(tree):

    root = None

    for current_node in tree:
        if current_node.depth == 0:
            root = current_node

    def traverse(this_node):

        if this_node.leaf:
            print(this_node)
            print("*********************************Path end*********************************")

        else:
            print(this_node)
            traverse(this_node.left_node)
            traverse(this_node.right_node)

    if root is None:
        print('Error, cannot find root node')

    else:
        traverse(root)


def predict(features, tree):

    root = None

    for node in tree:
        if node.depth == 0:
            root = node

    if root is None:
        print('Error, cannot find root node')
        return

    current_node = root

    while not current_node.leaf:
        if features[current_node.feature] <= current_node.value:
            current_node = current_node.get_left()
        else:
            current_node = current_node.get_right()

    return current_node.decision


def show_confusion_matrix(con_mat):

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(con_mat, cmap='GnBu')
    fig.colorbar(cax)

    alpha = ['Room 1', 'Room 2', 'Room 3', 'Room 4']

    ax.set_xticklabels(['']+alpha)
    ax.set_yticklabels(['']+alpha)

    plt.xlabel('Prediction')
    ax.xaxis.set_label_position('top')
    plt.ylabel('True Label')

    for (i, j), z in np.ndenumerate(con_mat):
        ax.text(j, i, '{}'.format(int(z)), ha='center', va='bottom')

    plt.show()


def evaluate(test_ds, trained_tree, print_stats=True, show_mat=True):

    confusion_mat = np.zeros((len(np.unique(test_ds[:, 7])), (len(np.unique(test_ds[:, 7])))))

    recall = []  # True positives / True positives + False negatives
    precision = []  # True positives / True positives + False positives

    F1 = []  # 2 * recall * precision / (recall + precision)

    score = 0

    for test_data in test_ds:

        if predict(test_data, trained_tree) == test_data[7]:

            score += 1

        confusion_mat[test_data[7] - 1, predict(test_data, trained_tree) - 1] += 1

    class_rate = score / len(test_ds)

    for room in range(len(np.unique(test_ds[:, 7]))):
        this_recall = confusion_mat[room, room] / sum(confusion_mat[:, room])
        this_precision = confusion_mat[room, room] / sum(confusion_mat[room])

        this_F1 = 2 * this_recall * this_precision / (this_recall + this_precision)

        recall.append(this_recall)
        precision.append(this_precision)
        F1.append(this_F1)


    if print_stats:

        print('')
        print('Classification rate: {}'.format(class_rate))

        print('''
            Recall         Precision           F1 
        ''')

        for room in np.unique(test_ds[:, 7]):

            print(
'''Room {}:      {:0.3f}           {:0.3f}           {:0.3f}'''.format(room, recall[room-1], precision[room-1], F1[room-1]))

    if show_mat:
        show_confusion_matrix(confusion_mat)

    return confusion_mat, class_rate, recall, precision, F1


def cross_validation(dataset, k=10, pruning=False, print_stats=True, show_mat=True):

    print('\nInitiating evaluation...\n')

    trees = []
    train_subgroups = []
    validation_subgroups = []
    test_subgroups = []

    total_confusion_mat = np.zeros((len(np.unique(dataset[:, 7])), (len(np.unique(dataset[:, 7])))))
    avg_recall = [0] * len(np.unique(dataset[:, 7]))
    avg_precision = [0] * len(np.unique(dataset[:, 7]))
    avg_F1 = [0] * len(np.unique(dataset[:, 7]))
    avg_class_rate = 0

    np.random.shuffle(dataset)

    for i in range(k):

        print('Cross validation: Fold {}'.format(i))

        train_set = np.delete(dataset, range(int((len(dataset)/k)*i), int((len(dataset)/k)*(i+1))), axis=0)



        if i+1 == k:
            validation_set = dataset[0: int((len(dataset) / k))]
            train_set = np.delete(train_set, range(0, int((len(dataset) / k))), axis=0)
        else:
            validation_set = dataset[int((len(dataset)/k)*i+1): int((len(dataset)/k)*(i+2))]
            train_set = np.delete(train_set, range(int((len(dataset) / k) * i+1), int((len(dataset) / k) * (i + 2))), axis=0)

        test_set = dataset[int((len(dataset)/k)*i): int((len(dataset)/k)*(i+1))]

        train_subgroups.append(train_set)
        test_subgroups.append(test_set)
        validation_subgroups.append(validation_set)

        current_tree = train(train_set)

        if pruning:
            current_tree = prune(current_tree, validation_set, print_info=False)

        trees.append(current_tree)

        con_mat, class_rate, recall, precision, F1 = evaluate(test_set, current_tree, print_stats=False, show_mat=False)

        total_confusion_mat = np.add(total_confusion_mat, con_mat)
        avg_recall = list(map(add, avg_recall, recall))
        avg_precision = list(map(add, avg_precision, precision))
        avg_F1 = list(map(add, avg_F1, F1))

        avg_class_rate += class_rate

    avg_recall = [x/k for x in avg_recall]
    avg_precision = [x/k for x in avg_precision]
    avg_F1 = [x/k for x in avg_F1]
    avg_class_rate = avg_class_rate/k

    print('\nEvaluation Completed\n')

    if print_stats:

        print('Average classification rate: {}'.format(avg_class_rate))

        print('''
                Average recall         Average precision           Average F1 
            ''')

        for room in np.unique(dataset[:, 7]):
            print(
'''Room {}:              {:0.3f}                   {:0.3f}                   {:0.3f}'''.format(room, avg_recall[room - 1],
                                                                                       avg_precision[room - 1],
                                                                                       avg_F1[room - 1]))
    if show_mat:
        show_confusion_matrix(total_confusion_mat)

    return total_confusion_mat, avg_class_rate, avg_recall, avg_precision, avg_F1


def prune(tree, test_ds, print_info=True):

    best_tree = tree
    pruned = True
    num_loop = 0

    while pruned:
        pruned = False
        num_loop += 1

        for ind in range(len(best_tree)):
            if not best_tree[ind].leaf or None:
                if best_tree[ind].left_node.leaf and best_tree[ind].right_node.leaf:
                    pre_prune_tree = copy.deepcopy(best_tree)
                    keep_left_tree = copy.deepcopy(best_tree)
                    keep_right_tree = copy.deepcopy(best_tree)

                    ##### pruning: keep left branch #####

                    node = keep_left_tree[ind]

                    node.leaf = True
                    node.decision = node.left_node.decision

                    node.feature = None
                    node.value = None
                    node.entropy_gain = 0
                    node.l_len = node.l_len + node.r_len
                    node.r_len = 0

                    node.left_node = None
                    node.right_node = None

                    node.left_node = None
                    node.right_node = None

                    ##### pruning: keep right branch #####

                    node = keep_right_tree[ind]

                    node.leaf = True
                    node.decision = node.right_node.decision

                    node.feature = None
                    node.value = None
                    node.entropy_gain = 0
                    node.l_len = node.l_len + node.r_len
                    node.r_len = 0

                    node.left_node = None
                    node.right_node = None

                    node.left_node = None
                    node.right_node = None

                    ##### Evaluate accuracies #####

                    _, pre_prune_class_rate, _, _, _ = evaluate(test_ds, pre_prune_tree, print_stats=False, show_mat=False)
                    _, keep_left_class_rate, _, _, _ = evaluate(test_ds, keep_left_tree, print_stats=False, show_mat=False)
                    _, keep_right_class_rate, _, _, _ = evaluate(test_ds, keep_right_tree, print_stats=False, show_mat=False)

                    if print_info:
                        print("******************************** Found node with 2 leaves *********************************")
                        print(best_tree[ind])
                        print('Pre-prune accuracy: {}'.format(pre_prune_class_rate))
                        print('Keep left accuracy: {}'.format(keep_left_class_rate))
                        print('Keep right accuracy: {}\n'.format(keep_right_class_rate))

                    if max(pre_prune_class_rate, keep_left_class_rate, keep_right_class_rate) == pre_prune_class_rate:
                        if print_info:
                            print('Node is not pruned')
                        best_tree = pre_prune_tree

                    elif max(pre_prune_class_rate, keep_left_class_rate, keep_right_class_rate) == keep_left_class_rate:
                        if print_info:
                            print('Node is pruned, adopted left leaf decision: {}'.format(best_tree[ind].left_node.decision))
                        best_tree = keep_left_tree
                        pruned = True

                    elif max(pre_prune_class_rate, keep_left_class_rate, keep_right_class_rate) == keep_right_class_rate:
                        if print_info:
                            print('Node is pruned, adopted right leaf decision: {}'.format(best_tree[ind].right_node.decision))
                        best_tree = keep_right_tree
                        pruned = True

    return best_tree








