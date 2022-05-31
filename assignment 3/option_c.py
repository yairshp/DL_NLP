import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils


class OptionCDataset(Dataset):
    def __init__(self, data_path, tags, is_train, corpus, prefixes, suffixes, delimiter=' '):
        self.is_train = is_train
        self.corpus = corpus
        self.prefixes = prefixes
        self.suffixes = suffixes
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
                word = raw_data[0][i].lower()
                tag = raw_data[1][i]
                prefix = utils.get_prefix(word)
                suffix = utils.get_suffix(word)
                prefix_index = self.prefixes[prefix] if prefix in self.prefixes else self.corpus[utils.UNKNOWN]
                suffix_index = self.suffixes[suffix] if suffix in self.suffixes else self.corpus[utils.UNKNOWN]
                word_index = self.corpus[word] if word in self.corpus else self.corpus[utils.UNKNOWN]
                tag_index = self.tags.index(tag)
                sentence_words.append((prefix_index, word_index, suffix_index))
                sentence_tags.append(tag_index)
            else:
                word = raw_data[0][i]
                word_index = self.corpus[word] if word in self.corpus else self.corpus[utils.UNKNOWN]
                sentence_words.append(word_index)
        return sentences

    def sentence_to_tensor(self, sentence_words, sentence_tags):
        words_tenors = [torch.tensor(indices) for indices in sentence_words]
        if self.is_train:
            sentence_tags_one_hot = np.zeros((len(sentence_tags), len(self.tags)))
            for i, tag in enumerate(sentence_tags):
                sentence_tags_one_hot[i][tag] = 1.
            sentence_tags = torch.tensor(sentence_tags_one_hot)
            sentence = (words_tenors, sentence_tags)
        else:
            sentence = words_tenors
        return sentence


class Lstm(nn.Module):
    def __init__(self, input_dim, hidden_state_dim, is_forward=True,
                 is_embedding_layer=True, embeddings_matrix=None, num_of_prefixes=None, num_of_suffixes=None):
        super(Lstm, self).__init__()

        self.is_forward = is_forward
        self.is_embedding_layer = is_embedding_layer

        self.input_dim = input_dim
        self.hidden_state_dim = hidden_state_dim

        if self.is_embedding_layer:
            self.words_embeddings_layer = nn.Embedding.from_pretrained(embeddings_matrix)
            self.prefix_embeddings = nn.Embedding(num_of_prefixes, input_dim)
            self.suffix_embeddings = nn.Embedding(num_of_suffixes, input_dim)

        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_state_dim)

    def forward(self, x):
        sequence_len = len(x) if self.is_embedding_layer else x.size(0)

        hidden_state = torch.zeros(1, self.hidden_state_dim)
        cell_state = torch.zeros(1, self.hidden_state_dim)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)

        out = x

        if self.is_embedding_layer:
            lstm_input = []
            for w in out:
                word_embedding = self.words_embeddings_layer(w[1])
                prefix_embedding = self.prefix_embeddings(w[0])
                suffix_embedding = self.suffix_embeddings(w[2])
                w = torch.add(word_embedding, prefix_embedding)
                w = torch.add(w, suffix_embedding)
                lstm_input.append(w)
            out = torch.stack(lstm_input)

        iterating_range = range(sequence_len) if self.is_forward else range(sequence_len - 1, -1, -1)
        hidden_states = []
        for i in iterating_range:
            hidden_state, cell_state = self.lstm_cell(out[i], (hidden_state, cell_state))
            hidden_states.append(hidden_state)
        return torch.stack(hidden_states)


def main():
    corpus, prefixes, suffixes = utils.create_corpus_with_subwords(f'{utils.EMBEDDING_PATH}/vocab.txt')
    embedding_matrix = utils.create_embedding_matrix(f'{utils.EMBEDDING_PATH}/wordVectors.txt')
    train_dataset = OptionCDataset(utils.POS_DEBUG_PATH, utils.POS_TAGS,
                                   is_train=True, corpus=corpus, prefixes=prefixes, suffixes=suffixes)
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=False)
    model = Lstm(50, 60, num_of_prefixes=len(prefixes)+1, num_of_suffixes=len(suffixes)+1,
                 embeddings_matrix=embedding_matrix)
    for x, y in train_dataloader:
        outputs = model(x)
        pass


if __name__ == '__main__':
    main()
