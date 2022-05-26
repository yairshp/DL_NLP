import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

POS_DEBUG_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/debug'
POS_TRAIN_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/train'
POS_DEV_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/dev'
POS_TEST_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/test'
UNKNOWN = 'UUUNNNKKK'
POS_TAGS = ['PADDING', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
            'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', ',', '.', '\'\'', '(', ')', ':', '$', '#']
EMBEDDING_DIM = 50
HIDDEN_STATE_DIM_1 = 100
HIDDEN_STATE_DIM_2 = 200
HIDDEN_LAYER_DIM = 1000
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001


class PosDataset(Dataset):
    def __init__(self, data_path, corpus, tags, is_train=True):
        self.is_train = is_train

        self.corpus = corpus
        self.tags = tags

        raw_data = pd.read_csv(data_path, delimiter=' ', skip_blank_lines=False, header=None)
        self.sentences = self.extract_sentences(raw_data)
        # self.max_sentence_length = PosDataset.get_max_sentence_length(raw_data)
        self.max_sentence_length = 150
        self.sentences = self.pad_sentences()

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
            sentence_tags_one_hot = np.zeros((len(sentence_tags), len(POS_TAGS)))
            for i, tag in enumerate(sentence_tags):
                sentence_tags_one_hot[i][tag] = 1.
            sentence_tags = torch.tensor(sentence_tags_one_hot)
            sentence = (sentence_words, sentence_tags)
        else:
            sentence = sentence_words
        return sentence

    def pad_sentences(self):
        padded_sentences = []
        for sentence in self.sentences:
            difference = self.max_sentence_length - sentence[0].size(0)
            words_padding = torch.zeros(difference, dtype=torch.long)
            padded_words = torch.cat((sentence[0], words_padding))
            tags_padding = torch.zeros((difference, len(POS_TAGS)), dtype=torch.long)
            for t in tags_padding:
                t[0] = 1
            padded_tags = torch.vstack((sentence[1], tags_padding))
            padded_sentences.append((padded_words, padded_tags))
        return padded_sentences

    @staticmethod
    def get_max_sentence_length(data_df):
        max_length = 0
        counter = 0
        for i in range(len(data_df)):
            if type(data_df[0][i]) != str:  # if the word is not nan
                max_length = counter if counter > max_length else max_length
                counter = 0
            else:
                counter += 1
        return max_length


class Lstm(nn.Module):
    def __init__(self, embedding_dim, hidden_state_dim, corpus_size, sequence_length, is_forward,
                 is_embedding_layer=True):
        super(Lstm, self).__init__()

        self.is_forward = is_forward
        self.is_embedding_layer = is_embedding_layer
        self.sequence_length = sequence_length

        self.embedding_dim = embedding_dim
        self.hidden_state_dim = hidden_state_dim

        if self.is_embedding_layer:
            self.embedding = nn.Embedding(corpus_size, self.embedding_dim, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(self.embedding_dim, self.hidden_state_dim)

    def forward(self, x):
        batch_size = x.size(0)

        hidden_state = torch.zeros(batch_size, self.hidden_state_dim)
        cell_state = torch.zeros(batch_size, self.hidden_state_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        out = x  # todo check if right

        if self.is_embedding_layer:
            out = self.embedding(out)
            # out = out.view(self.sequence_length, batch_size, -1)

        iterating_range = range(self.sequence_length) if self.is_forward else range(self.sequence_length - 1, -1, -1)
        hidden_states = [[] for i in range(batch_size)]
        for i in iterating_range:
            hidden_state, cell_state = self.lstm_cell(out[:, i, :], (hidden_state, cell_state))
            for j in range(batch_size):
                hidden_states[j].append(hidden_state[j])
        hidden_states = torch.stack([torch.stack(h) for h in hidden_states])
        return hidden_states


class PosTaggerRnn(nn.Module):
    def __init__(self, corpus_size, original_sequence_length):
        super(PosTaggerRnn, self).__init__()

        self.forward_lstm_1 = Lstm(EMBEDDING_DIM, HIDDEN_STATE_DIM_1, corpus_size, original_sequence_length,
                                   is_forward=True)
        self.backward_lstm_1 = Lstm(EMBEDDING_DIM, HIDDEN_STATE_DIM_1, corpus_size, original_sequence_length,
                                    is_forward=False)
        self.forward_lstm_2 = Lstm(2 * HIDDEN_STATE_DIM_1, HIDDEN_STATE_DIM_2, corpus_size,
                                   original_sequence_length, is_forward=True, is_embedding_layer=False)
        self.backward_lstm_2 = Lstm(2 * HIDDEN_STATE_DIM_1, HIDDEN_STATE_DIM_2, corpus_size,
                                    original_sequence_length, is_forward=False, is_embedding_layer=False)
        self.linear1 = nn.Linear(2 * HIDDEN_STATE_DIM_2, HIDDEN_LAYER_DIM)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(HIDDEN_LAYER_DIM, len(POS_TAGS))

    def forward(self, x):
        forward_1_output = self.forward_lstm_1(x)
        backward_1_output = self.backward_lstm_1(x)
        layer_1_output = torch.cat((forward_1_output, backward_1_output), dim=2)

        forward_2_output = self.forward_lstm_2(layer_1_output)
        backward_2_output = self.backward_lstm_2(layer_1_output)
        layer_2_output = torch.cat((forward_2_output, backward_2_output), dim=2)

        out = self.linear1(layer_2_output)
        out = self.tanh(out)
        out = self.linear2(out)

        return out


def create_corpus(corpus_path):
    raw_data = pd.read_csv(corpus_path, delimiter=' ', header=None)
    words = raw_data[0]
    words_counter = 1
    corpus = {}
    for word in words:
        if word in corpus:
            continue
        corpus[word] = words_counter
        words_counter += 1
    corpus[UNKNOWN] = words_counter
    return corpus


def train(corpus_size, original_sequence_length, train_dataloader, dev_dataloader):
    model = PosTaggerRnn(corpus_size, original_sequence_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in train_dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        accuracy = validate(model, dev_dataloader)
        # accuracy = 0
        print(f'epoch {epoch + 1}: loss - {epoch_loss / len(train_dataloader)}, accuracy - {accuracy}')


def validate(model, data):
    with torch.no_grad():
        correct = 0
        samples = 0

        for x, y in data:
            outputs = model(x)
            _, predictions = torch.max(outputs, 2)
            pass
            for y_i, p_i in zip(y, predictions):
                for w, tag in zip(y_i, p_i):
                    if w[tag.item()] == 1:
                        correct += 1
                    samples += 1

        return correct / samples


def main():
    corpus = create_corpus(POS_TRAIN_PATH)
    train_data = PosDataset(POS_DEBUG_PATH, corpus, POS_TAGS)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    dev_data = PosDataset(POS_DEV_PATH, corpus, POS_TAGS)
    dev_dataloader = DataLoader(dataset=dev_data, batch_size=BATCH_SIZE, shuffle=False)
    train(len(corpus) + 1, train_data.max_sentence_length, train_dataloader, dev_dataloader)


if __name__ == '__main__':
    main()

