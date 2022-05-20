import numpy as np
import re
from collections import Counter

STUDENT = {'name': 'Yair Shpitzer',
           'ID': '313285942'}

TRAIN_PATH = '../data/train'
DEV_PATH = '../data/dev'
CHARACTERS_TO_REPLACE = '[0-9!?*(){}@:,#&.=%$^;+_/\\-\'\"]'
NUM_ITERATIONS = 1000
LEARNING_RATE = 2*1e-5
SIZE_OF_VOCAB = 300


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.

    max_of_each_row = np.max(x)
    e_x = np.exp(x - max_of_each_row)
    sum_of_e_x = np.sum(e_x)
    x = e_x / sum_of_e_x
    return x


def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag = params
    probs = softmax(np.dot(np.tanh(np.dot(x, W) + b), U) + b_tag)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params
    y_hat = classifier_output(x, params)
    y_hat += 1e-10
    y_encoded = np.zeros(y_hat.shape)
    y_encoded[y] = 1

    z1 = np.dot(x, W) + b
    h1 = np.tanh(z1)
    z2 = np.dot(h1, U) + b_tag
    h2 = softmax(z2)
    loss = -np.log(y_hat[y])

    gL = y_hat - y_encoded
    gU = np.dot(h1.reshape(-1, 1), gL.reshape(1, -1))
    gb_tag = gL * 1
    tanh_derivative = 1 - h1 ** 2
    gW = np.dot(x.reshape(-1, 1), (np.dot(U, gL.reshape(-1, )) * tanh_derivative).reshape(1, -1))
    gb = np.dot(U, gL.reshape(-1, )) * tanh_derivative
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(in_dim, hid_dim)
    b = np.random.randn(hid_dim)
    U = np.random.randn(hid_dim, out_dim)
    b_tag = np.random.randn(out_dim)
    params = [W, b, U, b_tag]
    return params

