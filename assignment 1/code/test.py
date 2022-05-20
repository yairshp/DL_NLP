# import loglinear as ll
import mlp1
import random
from collections import Counter
import re
import numpy as np

STUDENT = {'name': 'Yair Shpitzer',
           'ID': '313285942'}

TRAIN_PATH = '../data/train'
DEV_PATH = '../data/dev'
TEST_PATH = '../data/test'
CHARACTERS_TO_REPLACE = '[0-9!?*(){}@:,#&.=%$^;+_/\\-\'\"]'
NUM_ITERATIONS = 100
LEARNING_RATE = 0.1
SIZE_OF_VOCAB = 600


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    bigrams = list(features.keys())
    bigrams.sort()
    result = [features[bigram] for bigram in bigrams]
    return np.array(result)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        features_vec = feats_to_vec(features)
        y_hat = mlp1.predict(features_vec, params)
        if y_hat == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            params[0] -= learning_rate * grads[0]
            params[1] -= learning_rate * grads[1]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def load_data(train_path, dev_path, test_path):
    train_data = []
    with open(train_path, 'r', encoding='utf8') as train_file:
        lines = train_file.readlines()
        for line in lines:
            label, text = line.strip().lower().split("\t", 1)
            text = re.sub(CHARACTERS_TO_REPLACE, '', text)
            train_data.append((label, text))
    dev_data = []
    with open(dev_path, 'r', encoding='utf8') as dev_file:
        lines = dev_file.readlines()
        for line in lines:
            label, text = line.strip().lower().split("\t", 1)
            text = re.sub(CHARACTERS_TO_REPLACE, '', text)
            dev_data.append((label, text))
    test_data = []
    with open(test_path, 'r', encoding='utf8') as test_file:
        lines = test_file.readlines()
        for line in lines:
            label, text = line.strip().lower().split("\t", 1)
            text = re.sub(CHARACTERS_TO_REPLACE, '', text)
            test_data.append((label, text))
    return train_data, dev_data, test_data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def create_features(vocab, L2I, F2I, bigrams, is_test):
    features = []
    for l, text in bigrams:
        line_bigrams = {F2I[bigram]: 0 for bigram in vocab}
        for bigram in text:
            if bigram not in vocab:
                continue
            line_bigrams[F2I[bigram]] += 1
        if is_test:
            features.append(line_bigrams)
        else:
            features.append((L2I[l], line_bigrams))
    return features


def preprocess_data(train_data, dev_data, test_data):
    train_bigrams = [(language, text_to_bigrams(text)) for language, text in train_data]
    dev_bigrams = [(language, text_to_bigrams(text)) for language, text in dev_data]
    test_bigrams = [(language, text_to_bigrams(text)) for language, text in test_data]

    fc = Counter()
    for l, feats in train_bigrams:
        fc.update(feats)
    t = Counter({bigram: count for bigram, count in fc.items() if len(bigram) > 1 and " " not in bigram})
    vocab = set([x for x, c in t.most_common(SIZE_OF_VOCAB)])
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}
    F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}

    train_features = create_features(vocab, L2I, F2I, train_bigrams, False)
    dev_features = create_features(vocab, L2I, F2I, dev_bigrams, False)
    test_features = create_features(vocab, L2I, F2I, test_bigrams, True)
    in_dim = len(vocab)
    out_dim = len(L2I.keys())
    return train_features, dev_features, test_features, in_dim, out_dim, L2I


def create_predictions_file(test_data, params, L2I):
    languages = list(L2I.keys())
    ids = list(L2I.values())
    with open('test.pred', 'w') as pred_file:
        for features in test_data:
            y = mlp1.predict(feats_to_vec(features), params)
            pred_file.write(languages[ids.index(y)] + '\n')


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    train_data, dev_data, test_data = load_data(TRAIN_PATH, DEV_PATH, TEST_PATH)
    train_features, dev_features, test_features, in_dim, out_dim, L2I = preprocess_data(train_data, dev_data, test_data)
    train_data, dev_data = train_features, dev_features

    params = mlp1.create_classifier(in_dim, 30, out_dim)
    num_iterations = NUM_ITERATIONS
    learning_rate = LEARNING_RATE
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    create_predictions_file(test_features, trained_params, L2I)





