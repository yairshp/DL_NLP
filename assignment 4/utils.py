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


def get_data_and_embeddings():
    x = torchtext.data.Field(init_token='</begin>', lower=True, batch_first=True, include_lengths=True)
    y = torchtext.data.Field(sequential=False)

    train_data, dev_data, test_data = torchtext.datasets.SNLI.splits(text_field=x, label_field=y)

    x.build_vocab(train_data, min_freq=1, vectors='glove.6B.300d')
    y.build_vocab(train_data)

    embedding_matrix = x.vocab.vectors

    train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits((train_data, dev_data, test_data),
                                                                           batch_size=BATCH_SIZE,
                                                                           sort_within_batch=True)

    return train_iter, dev_iter, test_iter, embedding_matrix


