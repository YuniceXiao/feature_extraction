#!/usr/bin/env python
import argparse
from util import load_reviews, shuffle_dataset
import numpy as np


def main(data_file):
    """extract function word features from a text file"""

    # TODO: parse the review file. Field [0] per line is the review ID. Field[-1] is the review
    reviews, ids = load_reviews(data_file)

    # debug using just a few
    reviews = reviews[:10]
    ids = ids[:10]

    feature_key = ["the", "or", "and"]

    print(f"loading feature vectors for {len(reviews)} reviews")
    feature_lists = []
    feature_np = np.zeros(((len(reviews)), len(feature_key)), dtype=np.int)

    lab_instructions = """ For function words "the", "or" and "and", use a Python list to 
    make a count vector per review
    Cast using np.asarray()
    Create a numpy array using np.zeros((num_reviews, num_function_words))
    Populate it by iterating over reviews, then iterating over function words
        for i,review in enumerate(reviews):
            for j,function_word in enumerate(function_words):
                feature_matrix[i,j] = ...
    Verify your numpy and lists are consistent"""

    print(f"instructions: {lab_instructions}")
    print(feature_lists)
    print(feature_np)


    lab_instructions2 = """Shuffle the list of id's and the feature matrix in unison. Then check your work"""

    print(f"instructions: {lab_instructions2}")

    print(f"Shuffling data")
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
