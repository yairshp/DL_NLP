import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

POS_DEBUG_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/debug'
UNKNOWN = 'UUUNNNKKK'
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
            'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', ',', '.', '\'\'', '(', ')', ':', '$', '#']
EMBEDDING_DIM = 50
WEIGHTS_DIM = 20


class PosDataset(Dataset):
    def __init__(self, data_path, corpus, tags, is_train=True):
        self.is_train = is_train

        self.corpus = corpus
        self.tags = tags

        raw_data = pd.read_csv(data_path, delimiter=' ', skip_blank_lines=False, header=None)
        self.sentences = self.extract_sentences(raw_data)
        # todo add padding

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        if self.is_train:
            return self.sentences[i][0], self.sentences[i][1]
        else:
            return self.sentences[i]

    def extract_sentences(self, data_df):
        print("extracting sentences...")
        sentences = []
        sentence_words = []
        sentence_tags = []
        for i in range(len(data_df)):
            if type(data_df[0][i]) != str:  # if the word is not nan
                sentence = self.sentence_to_tensor(sentence_words, sentence_tags)
                sentences.append(sentence)
                sentence_words = []
                sentence_tags = []
            elif self.is_train:
                word = data_df[0][i]
                tag = data_df[1][i]
                word_index = self.corpus[word] if word in self.corpus else self.corpus[UNKNOWN]
                tag_index = self.tags.index(tag)
                sentence_words.append(word_index)
                sentence_tags.append(tag_index)
            else:
                word = data_df[0][i]
                sentence_words.append(word)
        return sentences

    def sentence_to_tensor(self, sentence_words, sentence_tags):
        sentence_words = torch.tensor(sentence_words)
        if self.is_train:
            sentence_tags = torch.tensor(sentence_tags)
            sentence = (sentence_words, sentence_tags)
        else:
            sentence = sentence_words
        return sentence


class Lstm(nn.Module):
    def __init__(self, embedding_dim, weights_dim, corpus_size, is_forward, is_embedding_layer=True):
        super(Lstm, self).__init__()

        self.is_forward = is_forward
        self.is_embedding_layer = is_embedding_layer

        self.embedding_dim = embedding_dim
        self.weights_dim = weights_dim

        if self.is_embedding_layer:
            self.embedding = nn.Embedding(corpus_size, self.embedding_dim)
        self.lstm_cell = nn.LSTMCell(self.embedding_dim, self.weights_dim)

    def forward(self, x):
        # todo to batches? if yes, change also 'view' line
        sequence_len = x.size()[0]

        hidden_state = torch.zeros(1, self.weights_dim)
        cell_state = torch.zeros(1, self.weights_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        out = x  # todo check if right

        if self.is_embedding_layer:
            out = self.embedding(out)
            out = out.view(sequence_len, 1, -1)

        iterating_range = range(sequence_len) if self.is_forward else range(sequence_len - 1, -1, -1)
        for i in iterating_range:
            hidden_state, cell_state = self.lstm_cell(out[i], (hidden_state, cell_state))

        return hidden_state


class PosTaggerRnn(nn.Module):
    def __init__(self, corpus_size):
        super(PosTaggerRnn, self).__init__()

        self.forward_lstm_1 = Lstm(EMBEDDING_DIM, WEIGHTS_DIM, corpus_size, is_forward=True)
        self.backward_lstm_1 = Lstm(EMBEDDING_DIM, WEIGHTS_DIM, corpus_size, is_forward=False)

    def forward(self, x):
        forward_1_output = self.forward_lstm_1(x)
        backward_1_output = self.backward_lstm_1(x)

        return torch.cat([forward_1_output, backward_1_output], dim=1)


def create_corpus(corpus_path):
    raw_data = pd.read_csv(corpus_path, delimiter=' ', header=None)
    words = raw_data[0]
    words_counter = 0
    corpus = {}
    for word in words:
        if word in corpus:
            continue
        corpus[word] = words_counter
        words_counter += 1
    corpus[UNKNOWN] = words_counter
    return corpus


def main():
    corpus = create_corpus(POS_DEBUG_PATH)
    train_data = PosDataset(POS_DEBUG_PATH, corpus, POS_TAGS)
    dataloader = DataLoader(train_data, batch_size=None)
    model = PosTaggerRnn(len(corpus))
    for x, y in dataloader:
        model(x)


if __name__ == '__main__':
    main()

