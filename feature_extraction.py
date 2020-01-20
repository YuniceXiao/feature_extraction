#!/usr/bin/env python
import argparse
import numpy as np
from nltk import word_tokenize
from util import load_function_words, load_reviews, split_data


# do not alter this function
def check_splits(train, test, X, ids):
    """verify that all data is retained after splitting"""
    # check X
    train_X = train[0]
    val_X = test[0]
    sum_after = train_X.sum() + val_X.sum()
    assert sum_after == X.sum(), \
        f"Sum of features in Train+Test {sum_after} must equal sum of features before splitting {X.sum}"

    all_ids = train[1] + test[1]
    assert set(all_ids) == set(ids), "Set of ids in Train+Test must equal set of ids before splitting"
    # if we didn't crash, everything's good!
    print("Split checks passed!")


def main(data_file, vocab_path):
    """extract function word features from a text file"""

    # load resources and text file
    function_words = load_function_words(vocab_path)

    reviews, ids = load_reviews(data_file)

    # TODO: appropriately shape and fill this matrix
    nrows = len(ids)
    ncols = len(function_words)
    review_features = np.zeros((nrows, ncols), dtype=np.int)
    for i in range(len(reviews)):
        each_review = word_tokenize(reviews[i].lower())
        # each_review = reviews[i].split()
        for word in each_review:
            if word in function_words:
                word_index = function_words.index(word)
                review_features[i][word_index] += 1
    # row is which review
    # column is which word

    print(f"Numpy array has shape {review_features.shape} and dtype {review_features.dtype}")

    # TODO: Calculate these from review_features
    words_count = [sum(x) for x in zip(*review_features)]
    most_common_count = max(words_count)
    most_common_word_index = words_count.index(most_common_count)
    most_common_word = function_words[most_common_word_index]
    print(f"Most common word: {most_common_word}, count: {most_common_count}")

    # TODO: Find any features that weren't in the data (i.e. columns that sum to 0)
    zero_inds = []
    for i in range(len(words_count)):
        if words_count[i] == 0:
            zero_inds.append(i)

    if len(zero_inds) > 0:
        print("No instances found for: ")
        for ind in zero_inds:
            print(f"  {function_words[ind]}")
    else:
        print("All function words found")

    matrix_sum = review_features.sum()
    print(f"Sum of raw count matrix: {matrix_sum}")

    # TODO: make a binary feature vector from your count vector
    word_binary = np.copy(review_features)
    for i in range(len(word_binary)):
        for j in range(len(word_binary[i])):
            if word_binary[i][j] > 0:
                word_binary[i][j] = 1
    word_binary_sum = word_binary.sum()
    print(f"Sum of binary matrix: {word_binary_sum}")

    # TODO: normalize features by review length (divide rows by number of words in the review)
    norm_reviews = np.copy(review_features)
    norm_reviews = norm_reviews.tolist()
    for i in range(len(norm_reviews)):
        sum_of_row = sum(norm_reviews[i])
        for j in range(len(norm_reviews[i])):
            normalized_val = (norm_reviews[i][j]) / (sum_of_row)
            norm_reviews[i][j] = normalized_val
    norm_reviews = np.array(norm_reviews)
    norm_reviews_sum = round(norm_reviews.sum(), 2)
    print(f"Sum of normed matrix: {norm_reviews_sum}")

    # TODO: remove features from <review_features> that occur less than <min_count> times
    min_count = 100
    min_matrix = np.copy(review_features)
    remove_column_index = []
    for i in range(len(words_count)):
        if words_count[i] <= min_count:
            remove_column_index.append(i)
    min_matrix = np.delete(min_matrix, remove_column_index, 1)
    min_matrix_shape = min_matrix.shape
    print(f"Shape after removing features that occur < {min_count} times: {min_matrix_shape}")

    # TODO: split the dataset by updating the function above
    train, val = split_data(review_features, ids, 0.3)

    # Code below that all your data has been retained in your splits; do not edit.
    # Must all print True

    check_splits(train, val, review_features, ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="imdb_practice.txt",
                        help='path to input with one review per line')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
