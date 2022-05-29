import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils

EMBEDDING_DIM = 200
HIDDEN_STATE_DIM_1 = 300
HIDDEN_STATE_DIM_2 = 500
HIDDEN_LAYER_DIM = 2000
EPOCHS = 20
LEARNING_RATE = 0.001
MAX_WORD_LEN = 30


class OptionBDataset(Dataset):
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
            if utils.is_end_of_sentence(raw_data[0][i]):
                sentence = self.sentence_to_tensor(sentence_words, sentence_tags)
                sentences.append(sentence)
                sentence_words = []
                sentence_tags = []
            elif self.is_train:
                word = raw_data[0][i]
                tag = raw_data[1][i]
                char_indices = self.get_char_indices(word)
                tag_index = self.tags.index(tag)
                sentence_words.append(char_indices)
                sentence_tags.append(tag_index)
            else:
                word = raw_data[0][i]
                char_indices = self.get_char_indices(word)
                sentence_words.append(char_indices)
        return sentences

    def sentence_to_tensor(self, sentence_words, sentence_tags):
        sentence_words = torch.stack([torch.tensor(h) for h in sentence_words])
        if self.is_train:
            sentence_tags_one_hot = np.zeros((len(sentence_tags), len(self.tags)))
            for i, tag in enumerate(sentence_tags):
                sentence_tags_one_hot[i][tag] = 1.
            sentence_tags = torch.tensor(sentence_tags_one_hot)
            sentence = (sentence_words, sentence_tags)
        else:
            sentence = sentence_words
        return sentence

    @staticmethod
    def get_char_indices(word):
        indices = []
        for ch in word:

            indices.append(utils.CHARS.index(ch))
        difference = MAX_WORD_LEN - len(word)
        padding = [len(utils.CHARS) - 1 for i in range(difference)]
        indices.extend(padding)
        return indices


class CharsLstm(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, num_of_chars):
        super(CharsLstm, self).__init__()

        self.input_dim = input_dim
        self.hidden_state_dim = hidden_state_dim

        self.embedding_layer = nn.Embedding(num_of_chars, self.input_dim)
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_state_dim)

    def forward(self, x):
        num_of_words = x.size(0)

        hidden_state = torch.zeros(num_of_words, self.hidden_state_dim)
        cell_state = torch.zeros(num_of_words, self.hidden_state_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        out = self.embedding_layer(x)
        out = out.view(num_of_words, -1, self.input_dim)

        for i in range(MAX_WORD_LEN):
            hidden_state, cell_state = self.lstm_cell(out[:, i, :], (hidden_state, cell_state))

        return hidden_state


class WordsLstm(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, is_forward=True):
        super(WordsLstm, self).__init__()
        self.is_forward = is_forward

        self.input_dim = input_dim
        self.hidden_state_dim = hidden_state_dim

        self.lstm_cell = nn.LSTMCell(input_dim, hidden_state_dim)

    def forward(self, x):
        sequence_len = x.size(0)

        hidden_state = torch.zeros(1, self.hidden_state_dim)
        cell_state = torch.zeros(1, self.hidden_state_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        out = x.view(sequence_len, 1, -1)

        iterating_range = range(sequence_len) if self.is_forward else range(sequence_len - 1, -1, -1)
        hidden_states = []
        for i in iterating_range:
            hidden_state, cell_state = self.lstm_cell(out[i], (hidden_state, cell_state))
            hidden_states.append(hidden_state)
        return torch.stack(hidden_states)

def main():
    corpus = utils.create_corpus(utils.POS_TRAIN_PATH)
    train_data = OptionBDataset(utils.POS_DEBUG_PATH, utils.POS_TAGS, is_train=True, corpus=corpus)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=False)
    model = CharsLstm(15, 50, len(utils.CHARS))
    model2 = WordsLstm(50, 100)
    for x, y in train_dataloader:
        output = model(x)
        output = model2(output)
    pass


if __name__ == '__main__':
    main()
