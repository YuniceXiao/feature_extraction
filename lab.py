#!/usr/bin/env python
import argparse
from util import load_reviews, shuffle_dataset
import numpy as np
from nltk import word_tokenize


def main(data_file):
    """extract function word features from a text file"""

    # TODO: parse the review file. Field [0] per line is the review ID. Field[-1] is the review
    # define this function in util.py
    reviews, ids = load_reviews(data_file)

    # debug using just a few
    reviews = reviews[:10]
    ids = ids[:10]

    feature_key = ["the", "or", "and"]

    print(f"loading feature vectors for {len(reviews)} reviews")

    # TODO: For function words "the", "or" and "and", use a Python list to
    #     make a count vector per review
    feature_lists = []
    for review in reviews:
        review_words = word_tokenize(review.lower())
        vec = []
        for word in feature_key:
            these_words = [w for w in review_words if w == word]
            vec.append(len(these_words))
        feature_lists.append(vec)

    print(feature_lists)

    # TODO: Create the same feature vectors as a numpy array
    feature_np = np.zeros(((len(reviews)), len(feature_key)), dtype=np.int)
    for i,review in enumerate(reviews):
        review_words = word_tokenize(review.lower())
        for j,word in enumerate(feature_key):
            these_words = [w for w in review_words if w == word]
            feature_np[i,j] = len(these_words)

    print(feature_np)

    # TODO: Cast your feature_lists to a numpy array and then verify it is equivalent to feature_np
    feature_lists_np = np.asarray(feature_lists)
    print(f'equal? {np.array_equal(feature_lists_np, feature_np)}')

    # TODO: Shuffle the list of id's and the feature matrix in unison. Then check your work
    print(f"Shuffling data")
    #TODO: define this function in util.py
    shuffled_feature_matrix, shuffled_ids = shuffle_dataset(feature_np, ids)
    print("ids before shuffle")
    print(ids)
    print("ids after shuffle")
    print(shuffled_ids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector lab')
    parser.add_argument('--path', type=str, default="imdb_practice.txt",
                        help='path to input with one review per line')

    args = parser.parse_args()

    main(args.path)
