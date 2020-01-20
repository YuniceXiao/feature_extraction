#!/usr/bin/env python
import numpy as np
from sklearn.utils import shuffle

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words

# TODO: write this function (lab)
def load_reviews(data_file):
    """
    Load a tsv of movie reviews, where field 0 is review id and field -1 is review
    :param data_file: data file path
    :return: two lists, (reviews, ids)
    """
    reviews = []
    ids = []
    with open(data_file, 'r') as df:
        for line in df:
            fields = line.strip().split("\t")
            reviews.append(fields[-1])
            ids.append(fields[0])
    return reviews,ids

# TODO: write this function (lab)
def shuffle_dataset(data, id_strs):
    """
    Shuffles a list of datapoints and their id's in unison
    :param data: iterable, each item a datapoint
    :param id_strs: iterable, each item an id
    :return: tuple (shuffled_data, shuffled_id_strs)
    """
    shuffled_data, shuffled_ids = shuffle(data, id_strs)
    return (shuffled_data, shuffled_ids)

# TODO: write this function (homework)
def split_data(X, file_ids, test_percent = 0.3, shuffle=True):
    """
    Splits dataset for supervised learning and evaluation
    :param X: iterable of features
    :param file_ids: iterable of file id's corresponding the features in X
    :param test_percent: percent data to
    :param shuffle:
    :return: two tuples, (X_train, file_ids_train), (X_test, file_ids_test)
    """
    if shuffle:
        X, file_ids = shuffle_dataset(X, file_ids)

    # edit this to split the dataset
    split_point = round(len(X)*test_percent)
    X_train = X[split_point:]
    X_test = X[:split_point]
    file_ids_train = file_ids[split_point:]
    file_ids_test = file_ids[:split_point]
    train = X_train, file_ids_train
    test = X_test, file_ids_test
    return train, test