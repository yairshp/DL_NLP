import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils

CHARS_EMBEDDING_DIM = 10
CHARS_HIDDEN_DIM = 50
WORDS_HIDDEN_DIM_1 = 30
WORDS_HIDDEN_DIM_2 = 50
HIDDEN_LAYER_DIM = 100
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_WORD_LEN = 30
POS = 'POS'
NER = 'NER'


class OptionBDataset(Dataset):
    def __init__(self, data_path, tags, is_train, corpus, delimiter=' '):
        self.is_train = is_train
        self.corpus = corpus
        self.tags = tags

        raw_data = pd.read_csv(data_path, delimiter=delimiter, skip_blank_lines=False, header=None, quoting=3)
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


class CharsLstmB(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, num_of_chars):
        super(CharsLstmB, self).__init__()

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


class WordsLstmB(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, is_forward=True):
        super(WordsLstmB, self).__init__()
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


class TaggerB(nn.Module):
    def __init__(self, tags):
        super(TaggerB, self).__init__()

        self.tags = tags

        self.chars_lstm = CharsLstmB(CHARS_EMBEDDING_DIM, CHARS_HIDDEN_DIM, len(utils.CHARS))
        self.forward_lstm_1 = WordsLstmB(CHARS_HIDDEN_DIM, WORDS_HIDDEN_DIM_1, is_forward=True)
        self.backward_lstm_1 = WordsLstmB(CHARS_HIDDEN_DIM, WORDS_HIDDEN_DIM_1, is_forward=False)
        self.forward_lstm_2 = WordsLstmB(2 * WORDS_HIDDEN_DIM_1, WORDS_HIDDEN_DIM_2, is_forward=True)
        self.backward_lstm_2 = WordsLstmB(2 * WORDS_HIDDEN_DIM_1, WORDS_HIDDEN_DIM_2, is_forward=False)
        # self.linear_1 = nn.Linear(2 * WORDS_HIDDEN_DIM_2, HIDDEN_LAYER_DIM)
        # self.tanh = nn.Tanh()
        self.linear = nn.Linear(2 * WORDS_HIDDEN_DIM_2, len(self.tags))

    def forward(self, x):
        word_embedding = self.chars_lstm(x)
        forward_1_output = self.forward_lstm_1(word_embedding)
        backward_1_output = self.backward_lstm_1(word_embedding)
        layer_1_output = torch.cat((forward_1_output, backward_1_output), dim=2)

        forward_2_output = self.forward_lstm_2(layer_1_output)
        backward_2_output = self.backward_lstm_2(layer_1_output)
        layer_2_output = torch.cat((forward_2_output, backward_2_output), dim=2)

        out = self.linear(layer_2_output)
        # out = self.tanh(out)
        out = nn.Dropout()(out)
        # out = self.linear_2(out)

        return out


def train(train_dataloader, tags, dev_dataloader, ner_or_pos):
    model = TaggerB(tags)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'epoch {epoch + 1}:')
        sentence_counter = 0
        epoch_loss = 0
        for x, y in train_dataloader:
            sentence_counter += 1
            outputs = model(x)
            loss = criterion(outputs[:, 0, :], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if sentence_counter % 500 == 0:
                accuracy = validate(model, dev_dataloader, ner_or_pos)
                print(f'\tsentence {sentence_counter}: dev accuracy - {accuracy}')

    return model


def validate(model, dev_dataloader, ner_or_pos):
    with torch.no_grad():
        correct = 0
        samples = 0

        for x, y in dev_dataloader:
            samples += x.size(0)
            outputs = model(x)
            _, predictions = torch.max(outputs, 2)
            for y_i, p_i in zip(y, predictions):
                if ner_or_pos == POS:
                    if y_i[p_i.item()].item() != 0:
                        correct += 1
                else:
                    if y_i[p_i.item()].item() == 0:
                        continue
                    if utils.NER_TAGS[p_i.item()] == 'O':
                        samples -= 1
                        continue
                    correct += 1

        return correct / samples


def train_option_b(train_file, model_file, ner_or_pos, dev_file, corpus_path):
    delimiter = ' ' if ner_or_pos == POS else '\t'
    tags = utils.POS_TAGS if ner_or_pos == POS else utils.NER_TAGS
    corpus = utils.create_corpus(corpus_path, delimiter=delimiter)
    train_data = OptionBDataset(train_file, tags, corpus=corpus, is_train=True, delimiter=delimiter)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=True)
    dev_data = OptionBDataset(dev_file, tags, corpus=corpus, is_train=True, delimiter=delimiter)
    dev_dataloader = DataLoader(dev_data, batch_size=None, shuffle=False)
    model = train(train_dataloader, tags, dev_dataloader, ner_or_pos)
    torch.save(model, model_file)


def predict_model_b(model_file, input_file, output_file, corpus_path, ner_or_pos):
    # delimiter = ' ' if ner_or_pos == POS else '\t'
    tags = utils.POS_TAGS if ner_or_pos == POS else utils.NER_TAGS
    corpus = utils.create_corpus(corpus_path)
    model = torch.load(model_file)
    test_data = OptionBDataset(input_file, tags, corpus=corpus, is_train=False)
    test_dataloader = DataLoader(test_data, batch_size=None, shuffle=False)
    predictions = []
    for x in test_dataloader:
        outputs = model(x)
        _, sequence_predictions = torch.max(outputs, 2)
        for p in sequence_predictions:
            predictions.append(tags[p.item()])
        predictions.append('')
    words = pd.read_csv(input_file, header=None, skip_blank_lines=False, delimiter=' ')
    with open(output_file, 'w') as writer:
        for w, p in zip(words[0], predictions):
            if type(w) != str:
                writer.write('\n')
            writer.write(f'{w} {p}\n')


def main():
    corpus = utils.create_corpus(utils.POS_TRAIN_PATH)
    train_data = OptionBDataset(utils.POS_DEBUG_PATH, utils.POS_TAGS, is_train=True, corpus=corpus)
    train_dataloader = DataLoader(train_data, batch_size=None, shuffle=True)
    dev_data = OptionBDataset(utils.POS_DEV_PATH, utils.POS_TAGS, is_train=True, corpus=corpus)
    dev_dataloader = DataLoader(dev_data, batch_size=None, shuffle=False)
    train(train_dataloader, utils.POS_TAGS, dev_dataloader)


if __name__ == '__main__':
    main()
