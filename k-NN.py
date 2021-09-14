# ----------------------------------------------------------------------------------------------------------------------
# Implementation of k-NN algorithm for Machine Learning
# Primary developer: Abu Bakar, University of Kentucky
#
# Using/enhancing/butchering code: Sean Taylor Thomas
# 9/2021
# stth223@uky.edu
# ----------------------------------------------------------------------------------------------------------------------

from math import *
from decimal import Decimal
import random
import re
import numpy as numpy


def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) ** Decimal(root_value), 6)


# calculate distance
def calculate_distance(row1, row2, distance_func=2):
    row1, row2, = row1[:-1], row2[:-1]  # removing labels
    # return float(p_root(sum(pow(abs(a - b), distance_func) for a, b in zip(row1, row2)), distance_func)) #euclidian dist
    return sum(abs(row1-row2) for row1, row2 in zip(row1,row2)) # manhattan distance


# Test distance function
#
# dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]
#
# row0 = dataset[0]
# distances = [calculate_distance(row0, row, 4) for row in dataset]
# print(distances)


def get_neighbors(train, test_row, num_neighbors=3, distance_func=2):
    # compute distances
    distances = [(train_row, calculate_distance(test_row, train_row, distance_func))
                 for train_row in train]
    # sort
    distances.sort(key=lambda tup: tup[1])
    # get top-k neighbors
    neighbors = [distances[i][0] for i in range(num_neighbors)]

    return neighbors


# testing get_neighbors func with top 3 neighbors
# neighbors = get_neighbors(dataset, row0, distance_func=3)
# print(neighbors)


def predict_classification(train, test_row, num_neighbors=3, distance_func=2):
    neighbors = get_neighbors(train, test_row, num_neighbors, distance_func)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# # testing prediction function
# prediction = predict_classification(dataset, row0)
# print('Expected %d, Got %d.' % (dataset[0][-1], prediction), '\n')

# importing
from random import seed
from random import randrange
from csv import reader


# don't understand where to get data
# def load_data(filename):
#     file = open('X_train', 'r')
#

def word_extraction(sentence):
    ignore = []
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def tokenize(sentences):
    words = []
    #        print(sentence)
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words


# create a bag of words!
def load_bag_of_words():
    dataset = []
    all_sentences = []
    with open('synthetic_reviews.txt') as file:
        all_sentences = file.readlines()
        all_sentences = [line.rstrip() for line in all_sentences]

    vocab = tokenize(all_sentences)  # put each line as an element in vector
    print("Word List: \n{0} \n".format(vocab))

    for sentence in all_sentences:
        words = word_extraction(sentence)
        bag_vector = [0] * len(vocab)
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
                    # print("{0}\n{1}\n".format(sentence, numpy.array(bag_vector)))
        # bag_vector.append((round(random.random()))) # classifier
        dataset.append(bag_vector)
    for i in dataset:
        for j in i:
            j = str(j)
    return dataset


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def load_synthetic_data(dimensions):
    """Creating synthetic data here."""
    num_rows = 115
    num_cols = dimensions
    dataset = list()
    for rows in range(num_rows):
        row = []
        for cols in range(num_cols):
            row.append(str(round(random.random() * 1000 - 1, 1)))
            row.append(str(int(round(random.random()))))
        dataset.append(row)
    return dataset


# string to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# string to integer
def str_column_to_int(dataset, column):
    unique = set([row[column] for row in dataset])
    lookup = {value: i for i, value in enumerate(unique)}

    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# rescale dataset columns to range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    """ Evaluate Accuracy"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def other_metrics(actual, predicted):
    """ Evaluate Precision, F1 Score, Recall"""
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i] and actual[i] >= 0:
            true_pos += 1
        elif actual[i] == predicted[i] and actual[i] < 0:
            true_neg += 1
        elif actual[i] != predicted[i] and predicted[i] <= 0:
            false_neg += 1
        elif actual[i] != predicted[i] and predicted[i] > 0:
            false_pos += 1
    precision = true_pos / (true_pos + false_pos) * 100.0
    recall = true_pos / (true_pos + false_neg) * 100.0
    f1_score = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1_score]


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    metrics = [0,0,0]
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        metrics[0] += other_metrics(actual, predicted)[0]
        metrics[1] += other_metrics(actual, predicted)[1]
        metrics[2] += other_metrics(actual, predicted)[2]  # calculating other metrics
        scores.append(accuracy)
    print('Mean Precision: %', metrics[0] / float(len(scores)))
    print('Mean Recall: %', metrics[1] / float(len(scores)))
    print('Mean F1 Score: %', metrics[2] / float(len(scores)))
    return scores


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors=3, distance_func=2):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors, distance_func)
        predictions.append(output)
    return predictions


# Test the kNN on the Iris dataset
val = input("Bag of words? y/n:")
seed(1)


if val == "y" or val == "yes":
    dataset = load_bag_of_words()
else:
    num_dimensions = int(input("How many dimensions for synthetic data?"))
    dataset = load_synthetic_data(num_dimensions)

# filename = 'iris.csv'
# dataset = load_csv(filename)
if (val == "no" or val == "n"):
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

# evaluate algorithm
n_folds = 5
num_neighbors = 5
distance_func = 2
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors, distance_func)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
