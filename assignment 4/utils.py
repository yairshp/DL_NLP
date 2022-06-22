import re
import numpy as np
import torch
import torchtext.legacy as torchtext

BATCH_SIZE = 128
EMBEDDING_DIM = 300
DROPOUT_VALUE = 0.25
LEARNING_RATE = 0.001
BI_LSTM_HIDDEN_DIM = 300
NUM_OF_CLASSES = 3
NUM_OF_BI_LSTM_LAYERS = 1
EPOCHS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_random_embedding(w):
    x = np.random.uniform(-0.05, 0.05, EMBEDDING_DIM)
    return torch.tensor(x)


def get_data_and_embeddings():
    print('begin...')
    x = torchtext.data.Field(init_token='</begin>', lower=True, batch_first=True, include_lengths=True)
    y = torchtext.data.Field(sequential=False)

    print('getting dataset...')
    train_data, dev_data, test_data = torchtext.datasets.SNLI.splits(text_field=x, label_field=y)

    print('differentiating inputs...')
    train_data = differentiate_input(train_data)
    dev_data = differentiate_input(dev_data)
    test_data = differentiate_input(test_data)

    print('building vocab...')
    x.build_vocab(train_data, min_freq=1, vectors='glove.6B.300d', unk_init=generate_random_embedding)
    y.build_vocab(train_data)

    embedding_matrix = x.vocab.vectors

    train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits((train_data, dev_data, test_data),
                                                                           batch_size=BATCH_SIZE,
                                                                           sort_within_batch=True)

    return train_iter, dev_iter, test_iter, embedding_matrix


def differentiate_input(data):
    for example in data:
        original_hypothesis = example.hypothesis
        original_hypothesis = [re.sub('[^A-Za-z0-9]', '', w) for w in original_hypothesis]
        original_premise = example.premise
        original_premise = [re.sub('[^A-Za-z0-9]', '', w) for w in original_premise]
        example.hypothesis = [w for w in original_hypothesis if w not in original_premise]
        example.premise = [w for w in original_premise if w not in original_hypothesis]
    return data

