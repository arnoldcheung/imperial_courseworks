from train_arnold import *
from evaluation_arnold import *

clean_data = np.loadtxt('wifi_db/clean_dataset.txt').astype(int)
noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt').astype(int)
random_data = np.loadtxt('wifi_db/random_dataset.txt').astype(int)


# dataset = clean_data
#
# np.random.shuffle(dataset)
# k = 10
#
# train_set = np.delete(dataset, range(0, int((len(dataset)/k))), axis=0)
# test_set = dataset[0: int((len(dataset)/k))]
#
#
# tree = train(train_set)
#
# new_tree = prune(tree, test_set, print_info=False)
#
# evaluate(test_set, tree)
#
# evaluate(test_set, new_tree)


# cross_validation(noisy_data)
#
# cross_validation(clean_data)

# tree = train(clean_data)
#
# evaluate(noisy_data, tree)
#
# print_tree(tree)

cross_validation(noisy_data, k=10, pruning=False, print_stats=True, show_mat=True)

cross_validation(noisy_data, k=10, pruning=True, print_stats=True, show_mat=True)




