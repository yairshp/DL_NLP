import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gen_exmaples import generate_sequences


VOCAB = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd']
WEIGHTS_DIM = 20
LEARNING_RATE = 0.001
HIDDEN_LAYER_DIM = 500
EPOCHS = 5


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item][0], self.sequences[item][1]


class RnnAcceptor(nn.Module):
    def __init__(self, weights_dim, embedding_dim, hidden_layer_dim, vocab_size):
        super(RnnAcceptor, self).__init__()

        self.weights_dim = weights_dim
        self.embedding_dim = embedding_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.vocab_size = vocab_size
        self.output_size = 2

        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(self.embedding_dim, self.weights_dim)
        self.linear1 = nn.Linear(weights_dim, self.hidden_layer_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.hidden_layer_dim, self.output_size)

    def forward(self, x):
        # x is the list of indices of the chars in the vocab
        # todo to batches?
        hidden_state = torch.zeros(1, self.weights_dim)
        cell_state = torch.zeros(1, self.weights_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        # out = self.embedding(x)
        out = x.view(len(x), -1, len(VOCAB)).float()

        for i in range(len(x)):
            hidden_state, cell_state = self.lstm_cell(out[i], (hidden_state, cell_state))

        out = self.linear1(hidden_state)
        out = self.tanh(out)
        out = self.linear2(out)

        return out


def single_sequence_to_tensor(sequence):
    indices = []
    for char in sequence:
        char_one_hot = np.zeros(len(VOCAB))
        char_one_hot[VOCAB.index(char)] = 1.
        indices.append(char_one_hot)
    indices = np.array(indices)
    return torch.tensor(indices).float()


def multiple_sequences_to_tensor(sequences, is_train=True):
    tensor_sequences = []
    if is_train:
        for sequence, is_pos in sequences:
            label = torch.tensor([1., 0.], ) if is_pos else torch.tensor([0., 1.])
            label = label.view(-1, 2)
            tensor_sequences.append((single_sequence_to_tensor(sequence), label))
    else:
        for sequence in sequences:
            tensor_sequences.append(single_sequence_to_tensor(sequence))
    return tensor_sequences


def add_label(sequences, is_pos):
    results = []
    for sequence in sequences:
        results.append((sequence, is_pos))
    return results


# def get_data(train_data_size, test_data_size):
#     positive_train_sequences = generate_sequences(True, train_data_size)
#     positive_train_sequences_labeled = add_label(positive_train_sequences, True)
#     positive_train_data = multiple_sequences_to_tensor(positive_train_sequences_labeled)
#     negative_train_sequences = generate_sequences(False, train_data_size)
#     negative_train_sequences_labeled = add_label(negative_train_sequences, False)
#     negative_train_data = multiple_sequences_to_tensor(negative_train_sequences_labeled)
#
#     train_data = positive_train_data + negative_train_data
#     random.shuffle(train_data)
#
#     positive_test_sequences = generate_sequences(True, test_data_size)
#     positive_test_sequences_labeled = add_label(positive_test_sequences, True)
#     positive_test_data = multiple_sequences_to_tensor(positive_test_sequences_labeled)
#     negative_test_sequences = generate_sequences(False, test_data_size)
#     negative_test_sequences_labeled = add_label(negative_test_sequences, False)
#     negative_test_data = multiple_sequences_to_tensor(negative_test_sequences_labeled)
#
#     test_data = positive_test_data + negative_test_data
#
#     return train_data, test_data

def get_raw_data(input_file_name):
    with open(input_file_name, 'r') as input_file:
        data = input_file.readlines()
    data = [w[:-1] for w in data]
    return data


def prepare_data():
    # train data
    # raw_positive_data = get_raw_data('../data/pos_examples')
    # raw_negative_data = get_raw_data('../data/neg_examples')
    # raw_positive_data = get_raw_data('../data/section_2/palindrom_pos_train')
    # raw_negative_data = get_raw_data('../data/section_2/not_palindroms_train')
    raw_positive_data = get_raw_data('../data/section_2/same_size_train')
    raw_negative_data = get_raw_data('../data/section_2/different_size_train')

    labeled_positive_data = add_label(raw_positive_data, is_pos=True)
    labeled_negative_data = add_label(raw_negative_data, is_pos=False)

    tensored_positive_data = multiple_sequences_to_tensor(labeled_positive_data, is_train=True)
    tensored_negative_data = multiple_sequences_to_tensor(labeled_negative_data, is_train=True)

    train_data = tensored_positive_data + tensored_negative_data
    random.shuffle(train_data)

    # dev data
    # raw_positive_data = get_raw_data('../data/pos_dev')
    # raw_negative_data = get_raw_data('../data/neg_dev')
    # raw_positive_data = get_raw_data('../data/section_2/palindrom_pos_test')
    # raw_negative_data = get_raw_data('../data/section_2/not_palindroms_test')
    raw_positive_data = get_raw_data('../data/section_2/same_size_test')
    raw_negative_data = get_raw_data('../data/section_2/different_size_test')

    labeled_positive_data = add_label(raw_positive_data, is_pos=True)
    labeled_negative_data = add_label(raw_negative_data, is_pos=False)

    tensored_positive_data = multiple_sequences_to_tensor(labeled_positive_data, is_train=True)
    tensored_negative_data = multiple_sequences_to_tensor(labeled_negative_data, is_train=True)

    dev_data = tensored_positive_data + tensored_negative_data

    return train_data, dev_data


def train(train_data, test_data):
    model = RnnAcceptor(WEIGHTS_DIM, len(VOCAB), HIDDEN_LAYER_DIM, len(VOCAB))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataloader = DataLoader(dataset=train_data, batch_size=None)
    test_dataloader = DataLoader(dataset=test_data, batch_size=None)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in train_dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        accuracy = validate(test_dataloader, model)
        print(f'epoch {epoch + 1}: {epoch_loss / len(train_data)}, {accuracy}')

    return model


def validate(test_data, model):
    with torch.no_grad():
        correct = 0
        samples = 0

        for x, y in test_data:
            outputs = model(x)

            _, predictions = torch.max(outputs, 1)
            samples += 1
            if y[0][predictions.item()].item() == 1:
                correct += 1

        return correct / samples


def main():
    train_data, test_data = prepare_data()
    model = train(train_data, test_data)
    accuracy = validate(test_data, model)
    print(accuracy)


if __name__ == '__main__':
    main()
