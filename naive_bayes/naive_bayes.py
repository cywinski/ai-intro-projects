import random
from naive_bayes.math_functions import mean, std_dev, pdf

def classify(dataset):
    classified = dict()
    for i in range(len(dataset)):
        data = dataset[i]
        class_name = data[-1]
        if class_name not in classified.keys():
            classified[class_name] = list()
        classified[class_name].append(data[:-1])

    return classified


"""
    Calculates mean, standard deviation and length for every column in dataset

"""
def mean_and_stddev(classified):
    results = dict()
    for key in classified.keys():
        columns = zip(*classified[key])
        for col in columns:
            if key not in results.keys():
                results[key] = list()
            results[key].append((mean(list(col)), std_dev(list(col)), len(list(col))))

    return results


"""
    Calculates probability that particular vector belongs to particular class

"""
def class_probability(vector, class_stats):
    # P(class|x1, x2, ..., xn) = max(P(class) * P(x1|class) * P(x2|class) * ... * P(xn|class))
    probabilities = dict()
    # Calculate sum of vectors in data
    total_vectors = 0
    for key in class_stats.keys():
        total_vectors += class_stats[key][0][2]

    sum_prob_classes = 0
    # Calculate P(class) for every class
    for key in class_stats.keys():
        probabilities[key] = class_stats[key][0][2] / total_vectors

        # Calculate P(xn|class)
        for col in range(len(vector) - 1):
            prob_col = pdf(vector[col], class_stats[key][col][0], class_stats[key][col][1])
            probabilities[key] *= prob_col
        sum_prob_classes += probabilities[key]

    # Scale probabilities
    for key in class_stats.keys():
        probabilities[key] = probabilities[key] / sum_prob_classes

    return probabilities


"""
    Picks class with highest probability for particular vector

"""
def pick_best_class(vector, class_stats):
    probabilities = class_probability(vector, class_stats)
    best_class = ""
    best_prob = -1
    for key, value in probabilities.items():
        if value > best_prob:
            best_prob = value
            best_class = key

    return best_class, best_prob, probabilities


"""
    Splits dataset for train and test data

"""
def split_dataset(dataset, percentage_of_train):
    num_of_train = int((percentage_of_train / 100) * len(dataset))
    shuffled = random.sample(dataset, len(dataset))
    train = shuffled[:num_of_train]
    test = shuffled[num_of_train:]
    return train, test


def predict_classes(test, stats):
    predictions = []
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    total_probs = dict()

    for i in range(len(test)):
        predicted_class, prob, probabilities = pick_best_class(test[i], stats)
        for i in classes:
            if i not in total_probs.keys():
                total_probs[i] = list()
            total_probs[i].append(probabilities[i])
        predictions.append((predicted_class, prob))

    return predictions, total_probs


def naive_bayes(dataset, percentage_of_train):
    
    train, test = split_dataset(dataset, percentage_of_train)
    classified = classify(train)
    stats = mean_and_stddev(classified) # {class_n: [(mean_nth_col, std_dev_nth_col, len(nth_col))]}
    predictions, probabilities = predict_classes(test, stats)
    return test, predictions, probabilities
