import tree_lib as tl
import numpy as np

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')


def display_data(dataset, name, pruning=False, cross_validate=False):

    if cross_validate:
        M, c, p, r, F, Dm, Dstd = tl.cross_validation(dataset, 10, prune=pruning, show_matrix=True)
    else:
        idx = 200

        np.random.shuffle(dataset)
        train = dataset[:idx]
        test = dataset[idx:]

        lr = tl.Learner(train, test)
        lr.learn(verbose=True)

        if pruning:
            lr.pruning()

        M, c, p, r, F = lr.evaluate()
        Dm = lr.model_depth()
        Dstd = 0

    print('----------------------------------------')
    print(name)
    print('----------------------------------------')
    print('Confusion Matrix:\n{}\n'.format(M))
    print('Class Rate:\n{}\n'.format(c))
    print('Precision:\n{}\n'.format(p))
    print('Recall:\n{}\n'.format(r))
    print('F1:\n{}\n'.format(F))
    print('Average depth:\n{} +\- {}'.format(Dm, Dstd))
    print('----------------------------------------\n')

#display_data(noisy_dataset, 'Noisy')
#
#display_data(clean_dataset, 'Clean')
#
#display_data(noisy_dataset, 'Noisy - Pruned', pruning=True)
#
#display_data(clean_dataset, 'Clean - Pruned', pruning=True)


display_data(noisy_dataset, 'Noisy - Cross validated', cross_validate=True)

display_data(clean_dataset, 'Clean - Cross validated', cross_validate=True)

display_data(noisy_dataset, 'Noisy - Pruned - Cross validated', pruning=True, cross_validate=True)

display_data(clean_dataset, 'Clean - Pruned - Cross validated', pruning=True, cross_validate=True)
