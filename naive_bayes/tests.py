from matplotlib import pyplot as plt
from csv import reader
from naive_bayes.naive_bayes import naive_bayes, classify
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import itertools


def read_csv(path):
    dataset = list()
    with open(path, 'r') as rf:
        csv_reader = reader(rf, delimiter=',')
        for row in csv_reader:
            # Convert chars to float
            for i in range(len(row) - 1):
                row[i] = float(row[i])
            dataset.append(row)
    
    return dataset


"""
    Test accuracy of predictions for test data

"""
def test_accuracy(test, predictions):
    good_predictions = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i][0]:
            good_predictions += 1

    return "{:.2f}".format((good_predictions / len(predictions)) * 100)

def test_splitting_impact(dataset):
    accuracy = [0] * 10
    percentages = [x for x in range(0, 100, 10)]
    
    i = 0
    while (i < 20):
        for perc in range(10, 100, 10):
            test, predictions, probabilities = naive_bayes(dataset, perc)
            accuracy[perc // 10] += float(test_accuracy(test, predictions))
        i += 1
    
    for i in range(len(accuracy)):
        accuracy[i] /= 20


    plt.style.use('seaborn')
    plt.scatter(percentages, accuracy, edgecolor='black', linewidth=0.5)
    plt.xlabel('percentage of train data [%]')
    plt.ylabel('accuracy of predictions [%]')
    plt.title('splitting impact')
    plt.show()


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

def make_confusion_matrix(dataset, perc):
    avg_matrix = np.zeros(shape=(3, 3), dtype=int)
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    i = 0
    while i < 20:
        test, predictions, probabilities = naive_bayes(dataset, perc)
        y_true = [i[-1] for i in test]
        y_pred = [i[0] for i in predictions]
        matrix = confusion_matrix(y_true, y_pred, labels=classes)
        avg_matrix += matrix
        i += 1

    avg_matrix //= 20

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

def test_parameters(dataset):
    classified = classify(dataset)
    sepal_l, sepal_w, petal_l, petal_w = get_measures(dataset)
    length = len(classified['Iris-setosa'])

    plt.style.use('seaborn')
        
    plt.scatter(sepal_l[:length], sepal_w[:length], c='blue', label='Iris-setosa', edgecolors='black', linewidths=0.5)
    plt.scatter(sepal_l[length:2*length], sepal_w[length:2*length], c='green', label='Iris-versicolor', edgecolors='black', linewidths=0.5)
    plt.scatter(sepal_l[2*length:], sepal_w[2*length:], c='red', label='Iris-virginica', edgecolors='black', linewidths=0.5)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title('sepal info')
    plt.legend()
    plt.show()

    plt.scatter(petal_l[:length], petal_w[:length], c='blue', label='Iris-setosa', edgecolors='black', linewidths=0.5)
    plt.scatter(petal_l[length:2*length], petal_w[length:2*length], c='green', label='Iris-versicolor', edgecolors='black', linewidths=0.5)
    plt.scatter(petal_l[2*length:], petal_w[2*length:], c='red', label='Iris-virginica', edgecolors='black', linewidths=0.5)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('petal info')
    plt.legend()
    plt.show()


def get_measures(dataset):
    sepal_l = []
    sepal_w = []
    petal_l = []
    petal_w = []
    for i in dataset:
        sepal_l.append(i[0])
        sepal_w.append(i[1])
        petal_l.append(i[2])
        petal_w.append(i[3])
    

    return sepal_l, sepal_w, petal_l, petal_w


def make_roc_curve(dataset, perc):
    test, predictions, probabilities = naive_bayes(dataset, perc)
    y_true = [i[-1] for i in test]
    y_pred = [i[1] for i in predictions]

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    y_true_dict = dict()
    y_pred_dict = dict()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Binarize the output
    for i in classes:
        tmp_true = [0] * (len(y_true))
        tmp_pred = [0] * (len(y_pred))
        for j in range(len(y_true) - 1):
            tmp_true[j] = 1 if y_true[j] == i else 0
            tmp_pred[j] = probabilities[i][j]
        y_true_dict[i] = tmp_true
        y_pred_dict[i] = tmp_pred
        fpr[i], tpr[i], _ = roc_curve(y_true_dict[i], y_pred_dict[i])
        roc_auc[i] = auc(fpr[i], tpr[i])   

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    
    plt.figure()
    lw = 2
    for i, color in zip(range(len(classes)), colors):

        plt.plot(fpr[classes[i]], tpr[classes[i]], color=color,
            lw=lw, label=f'ROC curve (area = %0.2f) for class {classes[i]}' % roc_auc[classes[i]])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for Iris dataset')
    plt.legend(loc="lower right")
    plt.show()
    


if __name__ == "__main__":
    dataset = read_csv('iris.csv')
    test_splitting_impact(dataset)