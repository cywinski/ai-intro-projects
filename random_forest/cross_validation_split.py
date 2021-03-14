import random

def cross_validation_split(dataset, n_folds):
    splitted = list()
    dataset_cp = dataset.copy()
    fold_size = len(dataset) // n_folds
    for i in range(n_folds):
        fold = list()
        for i in range(fold_size):
            choice = random.randrange(len(dataset_cp))
            fold.append(dataset_cp.pop(choice))
        splitted.append(fold)
    return splitted