import loglinear as ll
import mlp1
import random
from collections import Counter
import re
import numpy as np
import xor_data

STUDENT = {'name': 'Yair Shpitzer',
           'ID': '313285942'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return np.array(features)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        features_vec = feats_to_vec(features)
        # y_hat = mlp1.predict(features_vec, params)
        y_hat = ll.predict(features_vec, params)
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
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            # loss, grads = mlp1.loss_and_gradients(x, y, params)
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    data = xor_data.data
    # params = mlp1.create_classifier(2, 3, 2)
    params = ll.create_classifier(2, 2)
    trained_params = train_classifier(data, data, 2000, 0.1, params)
