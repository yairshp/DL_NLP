import numpy as np

STUDENT = {'name': 'Yair Shpitzer',
           'ID': '313285942'}


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
    input_vec = x
    for W, b in zip(params[0:-2:2], params[1:-2:2]):
        input_vec = np.tanh(np.dot(input_vec, W) + b)
    probs = softmax(np.dot(input_vec, params[-2]) + params[-1])
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    y_hat = classifier_output(x, params)
    y_hat += 1e-10
    y_encoded = np.zeros(y_hat.shape)
    y_encoded[y] = 1
    loss = -np.log(y_hat[y])

    layers = []
    input_vec = x
    for W, b in zip(params[0:-2:2], params[1:-2:2]):
        layer = {'input': input_vec, 'W': W, 'b': b}
        input_vec = np.tanh(np.dot(input_vec, W) + b)
        layer['output'] = input_vec
        layers.append(layer)
    output = softmax(np.dot(input_vec, params[-2]) + params[-1])
    layers.append({'input': input_vec, 'W': params[-2], 'b': params[-1], 'output': output})

    gL = y_hat - y_encoded
    layers[-1]['gW'] = np.dot(layers[-1]['input'].reshape(-1, 1), gL.reshape(1, -1))
    layers[-1]['gb'] = gL * 1
    layers[-1]['g'] = gL
    for i in reversed(range(0, len(layers) - 1)):
        tanh_derivative = 1 - layers[i + 1]['input'] ** 2
        layers[i]['g'] = np.dot(layers[i + 1]['W'], layers[i + 1]['g'].reshape(-1,)) * tanh_derivative
        layers[i]['gW'] = np.dot(layers[i]['input'].reshape(-1, 1), layers[i]['g'].reshape(1, -1))
        layers[i]['gb'] = layers[i]['g']
    grads = []
    for layer in layers:
        grads.append(layer['gW'])
        grads.append(layer['gb'])
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        W_i = np.random.randn(in_dim, out_dim)
        b_i = np.random.randn(out_dim)
        params.append(W_i)
        params.append(b_i)
    return params

