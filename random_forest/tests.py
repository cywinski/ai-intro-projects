from random_forest.decision_tree import DecisionTree
from numpy.core.fromnumeric import size
from random_forest.cross_validation_split import cross_validation_split
from random_forest.random_forest import RandomForest
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import itertools
import time

def read_csv(path):
    dataset = list()
    with open(path, 'r') as rf:
        for line in rf:
            line = " ".join(line.split())
            line = line.split(' ')
            dataset.append(line)
            convert_to_float(line)
    
    return dataset


def convert_to_float(line):
    for i in range(len(line) - 1):
        line[i] = float(line[i])


def test_performance(dataset, num_of_folds, num_of_trees, max_depth, min_node_size, num_of_attributes):
    folds = cross_validation_split(dataset, num_of_folds)
    performance = 0
    for i in range(num_of_folds):
        fold_performance = 0
        folds_cp = folds.copy()
        test = folds_cp.pop(i)
        train = [row for group in folds_cp for row in group]
        rand_forest = RandomForest(train, num_of_trees, max_depth, min_node_size, num_of_attributes)
        y_true = [row[-1] for row in test]
        y_pred = [rand_forest.predict(row) for row in test]
        for j in range(len(y_true)):
            if y_true[j] == y_pred[j]:
                fold_performance += 1
        performance += fold_performance / len(y_true)
    performance /= num_of_folds
    return float("{:.2f}".format(performance))


def sensitivity(matrix):
    sens_a = matrix[0][0] / sum(matrix[0])
    sens_b = matrix[1][1] / sum(matrix[1])
    sens_c = matrix[2][2] / sum(matrix[2])
    return sens_a, sens_b, sens_c 


def precision(matrix):
    prec_a = matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0])
    prec_b = matrix[1][1] / (matrix[0][1] + matrix[1][1] + matrix[2][1])
    prec_c = matrix[2][2] / (matrix[0][2] + matrix[1][2] + matrix[2][2])
    return prec_a, prec_b, prec_c


"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

"""
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def make_confusion_matrix(dataset, num_of_folds, num_of_trees, max_depth, min_node_size, num_of_attributes):
    avg_matrix = np.zeros(shape=(3, 3), dtype=int)
    classes = ['1', '2', '3']
    start_time = time.time()
    folds = cross_validation_split(dataset, num_of_folds)
    for i in range(num_of_folds):
        folds_cp = folds.copy()
        test = folds_cp.pop(i)
        train = [row for group in folds_cp for row in group]
        rand_forest = RandomForest(train, num_of_trees, max_depth, min_node_size, num_of_attributes)
        y_true = [row[-1] for row in test]
        y_pred = [rand_forest.predict(row) for row in test]
        matrix = confusion_matrix(y_true, y_pred, labels=classes)
        avg_matrix += matrix
        
    avg_matrix //= num_of_folds
    print(f"Time: {time.time() - start_time}")

    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(avg_matrix, classes=classes,
                      title='Confusion matrix')

    sens_a, sens_b, sens_c = sensitivity(avg_matrix)
    print(f"sensitivity_A={sens_a}, sensitivity_B={sens_b}, sensitivity_C={sens_c}")
    prec_a, prec_b, prec_c = precision(avg_matrix)
    print(f"precision_A={prec_a}, precision_B={prec_b}, precision_C={prec_c}")
    accuracy = (avg_matrix[0][0] + avg_matrix[1][1] + avg_matrix[2][2]) / (sum(avg_matrix[0]) + sum(avg_matrix[1]) + sum(avg_matrix[2]))
    print(f"accuracy={accuracy}")

    return avg_matrix


def get_all_attributes(dataset):
    area = list()
    perimeter = list()
    compactness = list()
    len_of_kernel = list()
    width_of_kernel = list()
    asymmetry_coefficient = list()
    len_of_kernel_groove = list()
    
    for row in dataset:
        area.append(row[0])
        perimeter.append(row[1])
        compactness.append(row[2])
        len_of_kernel.append(row[3])
        width_of_kernel.append(row[4])
        asymmetry_coefficient.append(row[5])
        len_of_kernel_groove.append(row[6])
    return area, perimeter, compactness, len_of_kernel, width_of_kernel, asymmetry_coefficient, len_of_kernel_groove


def plot_parameters(param1, param2, xlabel, ylabel, title):
    plt.style.use('seaborn')

    plt.scatter(param1[:70], param2[:70], c='blue', label='Kama', edgecolors='black', linewidths=0.5, s=100)
    plt.scatter(param1[70:2*70], param2[70:2*70], c='green', label='Rosa', edgecolors='black', linewidths=0.5, s=100)
    plt.scatter(param1[2*70:], param2[2*70:], c='red', label='Canadian', edgecolors='black', linewidths=0.5, s=100)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = read_csv('seeds_dataset.csv')
    area, perimeter, compactness, len_of_kernel, width_of_kernel, asymmetry_coefficient, len_of_kernel_groove = get_all_attributes(dataset)
    tree = DecisionTree(dataset, 5, 10, 2, [0, 1, 2, 3, 4, 5, 6, 7])
    tree.visualize()