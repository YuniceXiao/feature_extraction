#!/usr/bin/env python
import argparse


def main(data_file):
    """extract function word features from a text file"""

    max_reviews = 10
    reviews = []
    with open(data_file, 'r') as df:
        for line in df:
            fields = line.strip().split("\t")
            reviews.append(fields[-1])
            if len(reviews) >= max_reviews:  # just read a few to debug
                break

    feature_key = ["the", "or", "and"]

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

    # part 2

    lab_instructions2 = """for the first ten reviews in the dataset, load the authors (field 1 on each line)
        make a key, {author_id : class_index}. Start class index at 0
        create a np array y of shape (num_reviews,) as your data labels"""

    print(f"instructions: {lab_instructions2}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="imdb_practice.txt",
                        help='path to the menu to update')

    args = parser.parse_args()

    main(args.path)
