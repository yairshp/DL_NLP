import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils

EMBEDDING_DIM = 20
HIDDEN_STATE_DIM_1 = 30
HIDDEN_STATE_DIM_2 = 50
HIDDEN_LAYER_DIM = 300
EPOCHS = 20
LEARNING_RATE = 0.005


class OptionADataset(Dataset):
    def __init__(self, data_path, tags, is_train, corpus, delimiter=' '):
        self.is_train = is_train
        self.corpus = corpus
        self.tags = tags

        raw_data = pd.read_csv(data_path, delimiter=delimiter, skip_blank_lines=False, header=None)
        self.sentences = self.extract_sentences_from_raw_data(raw_data)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        if self.is_train:
            return self.sentences[i][0], self.sentences[i][1]
        else:
            return self.sentences[i]

    def extract_sentences_from_raw_data(self, raw_data):
        print('extracting sentences...')
        sentences = []
        sentence_words = []
        sentence_tags = []
        for i in range(len(raw_data)):
            word = raw_data[0][i]
            if utils.is_end_of_sentence(word):
                sentence = self.sentence_to_tensor(sentence_words, sentence_tags)
                sentences.append(sentence)
                sentence_words = []
                sentence_tags = []
            elif self.is_train:
                word = raw_data[0][i]
                tag = raw_data[1][i]
                word_index = self.corpus[word] if word in self.corpus else self.corpus[utils.UNKNOWN]
                tag_index = self.tags.index(tag)
                sentence_words.append(word_index)
                sentence_tags.append(tag_index)
            else:
                word_index = self.corpus[word] if word in self.corpus else self.corpus[utils.UNKNOWN]
                sentence_words.append(word_index)
        return sentences

    def sentence_to_tensor(self, sentence_words, sentence_tags):
        sentence_words = torch.tensor(sentence_words)
        if self.is_train:
            sentence_tags_one_hot = np.zeros((len(sentence_tags), len(self.tags)))
            for i, tag in enumerate(sentence_tags):
                sentence_tags_one_hot[i][tag] = 1.
            sentence_tags = torch.tensor(sentence_tags_one_hot)
            sentence = (sentence_words, sentence_tags)
        else:
            sentence = sentence_words
        return sentence


class Lstm(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, corpus_size, is_embedding_layer=True, is_forward=True):
        super(Lstm, self).__init__()

        self.is_forward = is_forward
        self.is_embedding_layer = is_embedding_layer

        self.input_dim = input_dim
        self.hidden_state_dim = hidden_state_dim

        if self.is_embedding_layer:
            self.embedding_layer = nn.Embedding(corpus_size, self.input_dim)

        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_state_dim)

    def forward(self, x):
        sequence_len = x.size(0)

        hidden_state = torch.zeros(1, self.hidden_state_dim)
        cell_state = torch.zeros(1, self.hidden_state_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        out = x

        if self.is_embedding_layer:
            out = self.embedding_layer(out)
            out = out.view(sequence_len, -1, self.input_dim)

        iterating_range = range(sequence_len) if self.is_forward else range(sequence_len - 1, -1, -1)
        hidden_states = []
        for i in iterating_range:
            hidden_state, cell_state = self.lstm_cell(out[i], (hidden_state, cell_state))
            hidden_states.append(hidden_state)
        return torch.stack(hidden_states)


class Tagger(nn.Module):
    def __init__(self, tags, corpus_size):
        super(Tagger, self).__init__()

        self.forward_lstm_1 = Lstm(EMBEDDING_DIM, HIDDEN_STATE_DIM_1, corpus_size)
        self.backward_lstm_1 = Lstm(EMBEDDING_DIM, HIDDEN_STATE_DIM_1, corpus_size, is_forward=False)
        self.forward_lstm_2 = Lstm(2 * HIDDEN_STATE_DIM_1, HIDDEN_STATE_DIM_2, corpus_size, is_embedding_layer=False)
        self.backward_lstm_2 = Lstm(2 * HIDDEN_STATE_DIM_1, HIDDEN_STATE_DIM_2, corpus_size,
                                    is_forward=False, is_embedding_layer=False)
        self.linear_1 = nn.Linear(2 * HIDDEN_STATE_DIM_2, HIDDEN_LAYER_DIM)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(HIDDEN_LAYER_DIM, len(tags))

    def forward(self, x):
        forward_1_output = self.forward_lstm_1(x)
        backward_1_output = self.backward_lstm_1(x)
        layer_1_output = torch.cat((forward_1_output, backward_1_output), dim=2)

        forward_2_output = self.forward_lstm_2(layer_1_output)
        backward_2_output = self.backward_lstm_2(layer_1_output)
        layer_2_output = torch.cat((forward_2_output, backward_2_output), dim=2)

        out = self.linear_1(layer_2_output)
        out = self.tanh(out)
        # out = nn.Dropout()(out)
        out = self.linear_2(out)

        return out


def train(train_dataloader, dev_dataloader, tags, corpus_size):
    model = Tagger(tags, corpus_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in train_dataloader:
            outputs = model(x)
            loss = criterion(outputs[:, 0, :], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        accuracy = validate(model, dev_dataloader)
        # accuracy = 0
        print(f'epoch {epoch + 1}: loss - {epoch_loss / len(train_dataloader)}, accuracy - {accuracy}')


def validate(model, dev_dataloader):
    with torch.no_grad():
        correct = 0
        samples = 0

        for x, y in dev_dataloader:
            outputs = model(x)
            _, predictions = torch.max(outputs, 2)
            for y_i, p_i in zip(y, predictions):
                if y_i[p_i.item()].item() != 0:
                    correct += 1
                samples += 1

        return correct / samples


def main():
    corpus = utils.create_corpus(utils.POS_TRAIN_PATH)
    train_data = OptionADataset(utils.POS_DEBUG_PATH, utils.POS_TAGS, corpus=corpus, is_train=True)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=True)
    dev_data = OptionADataset(utils.POS_DEV_PATH, utils.POS_TAGS, corpus=corpus, is_train=True)
    dev_dataloader = DataLoader(dev_data, batch_size=None)
    train(train_dataloader, dev_dataloader, utils.POS_TAGS, len(corpus) + 1)


if __name__ == '__main__':
    main()