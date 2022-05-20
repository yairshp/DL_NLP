import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

POS_DATASET_PATH = "/home/yair/Documents/University/Deep Learning for NLP/assignment 2/data/pos"
NER_DATASET_PATH = "/home/yair/Documents/University/Deep Learning for NLP/assignment 2/data/ner"
EMBEDDINGS_PATH = "/home/yair/Documents/University/Deep Learning for NLP/assignment 2/embeddings"
EMBEDDING_VECTOR_SIZE = 50
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
            'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', ',', '.', '\'\'', '(', ')', ':', '$', '#']
NER_TAGS = ['PER', 'LOC', 'ORG', 'TIME', 'O', 'MISC']

BEGIN_TOKEN_EMBEDDING = torch.rand(EMBEDDING_VECTOR_SIZE)
END_TOKEN_EMBEDDING = torch.rand(EMBEDDING_VECTOR_SIZE)


class TaggerDataset(Dataset):
    def __init__(self, dataset_path, embeddings, tags,
                 is_train=True, are_embeddings_random=True, delimiter=" ", move_to_lower=False):
        self.are_embeddings_random = are_embeddings_random
        self.is_train = is_train
        self.is_move_to_lower = move_to_lower
        self.tags = tags
        self.data_df = pd.read_csv(dataset_path, delimiter=delimiter, header=None, skip_blank_lines=False, quoting=3)
        self.sentences = self.extract_raw_data_and_sentences(self.data_df)
        self.embeddings = embeddings
        self.embedded_data = self.get_embedded_data()

    def __len__(self):
        return len(self.embedded_data)

    def __getitem__(self, index):
        # if self.is_train:
        x, y = self.embedded_data[index]
        return x, y
        # else:
        #     x = self.embedded_data[index]
        #     return x

    def get_embedded_data(self):
        print("creating embedded data...")
        embedded_data = []
        for sentence in self.sentences:
            for i in range(len(sentence)):
                w1 = self.get_word_embedding(sentence[i - 2][0]) if i >= 2 \
                    else BEGIN_TOKEN_EMBEDDING
                w2 = self.get_word_embedding(sentence[i - 1][0]) if i >= 1 \
                    else BEGIN_TOKEN_EMBEDDING
                w3 = self.get_word_embedding(sentence[i][0])
                w4 = self.get_word_embedding(sentence[i + 1][0]) if i < len(sentence) - 1 \
                    else END_TOKEN_EMBEDDING
                w5 = self.get_word_embedding(sentence[i + 2][0]) if i < len(sentence) - 2 \
                    else END_TOKEN_EMBEDDING
                x = torch.cat((w1, w2, w3, w4, w5), 0)
                if self.is_train:
                    tag = sentence[i][1]
                    tag_idx = self.tags.index(tag)
                    y = torch.zeros(len(self.tags))
                    y[tag_idx] = 1.
                    embedded_data.append((x, y))
                else:
                    y = torch.zeros(len(self.tags))
                    embedded_data.append((x, y))
        return embedded_data

    def get_word_embedding(self, word):
        if self.is_move_to_lower:
            word = word.lower()
        if word in self.embeddings:
            return self.embeddings[word]
        elif self.are_embeddings_random:
            return torch.rand(EMBEDDING_VECTOR_SIZE)
        else:
            return self.embeddings['UUUNKKK']

    @staticmethod
    def get_embeddings(data_df):
        print('getting word embeddings...')
        words = data_df[0]
        embeddings = {}
        counter = 0
        for word in words:
            if type(word) != str or word in embeddings:  # if the word is not nan, or it is already in dict
                counter += 1
                continue
            embeddings[word] = torch.rand(EMBEDDING_VECTOR_SIZE)
        print(counter)
        return embeddings

    def extract_raw_data_and_sentences(self, data_df):
        print("extracting sentences...")
        sentences = []
        sentence = []
        for i in range(len(data_df)):
            if type(data_df[0][i]) != str:  # if the word is not nan
                sentences.append(sentence)
                sentence = []
            elif self.is_train:
                sentence.append((data_df[0][i], data_df[1][i]))
            else:
                sentence.append(data_df[0][i])
        return sentences


class SubWordTaggerDataset(TaggerDataset):
    def get_embedded_data(self):
        embedded_data = []
        for i, word in enumerate(self.data_df[0]):
            if type(word) != str:
                continue
            prefix = suffix = ''
            if len(word) <= 3:
                prefix = suffix = word
            elif len(word) > 3:
                prefix = word[:3]
                suffix = word[-3:]
            prefix_embedding = self.get_word_embedding(prefix)
            suffix_embedding = self.get_word_embedding(suffix)
            word_embedding = self.get_word_embedding(word)
            x = torch.add(torch.add(prefix_embedding, suffix_embedding), word_embedding)
            if self.is_train:
                tag = self.data_df[1][i]
                tag_idx = self.tags.index(tag)
                y = torch.zeros(len(self.tags))
                y[tag_idx] = 1.
                embedded_data.append((x, y))
            else:
                y = torch.zeros(len(self.tags))
                embedded_data.append((x, y))
        return embedded_data


def get_existing_embeddings(vectors_path, vocab_path):
    embeddings = {}
    vectors = np.loadtxt(vectors_path)
    with open(vocab_path, 'r') as r:
        vocab = r.readlines()
    vocab = [w[:-1] for w in vocab]
    for word, vector in zip(vocab, vectors):
        embeddings[word] = torch.from_numpy(vector)
    embeddings[' '] = vectors[-1]
    return embeddings


def get_existing_embeddings_with_subwords(vectors_path, vocab_path):
    embeddings = {}
    vectors = np.loadtxt(vectors_path)
    with open(vocab_path, 'r') as r:
        vocab = r.readlines()
    vocab = [w[:-1] for w in vocab]
    for word, vector in zip(vocab, vectors):
        if len(word) > 3:
            prefix = word[:3]
            suffix = word[-3:]
            if prefix not in embeddings:
                embeddings[prefix] = torch.rand(EMBEDDING_VECTOR_SIZE)
            if suffix not in embeddings:
                embeddings[suffix] = torch.rand(EMBEDDING_VECTOR_SIZE)
        embeddings[word] = torch.from_numpy(vector)
    embeddings[' '] = vectors[-1]
    return embeddings


def create_random_embeddings(data_df):
    words = data_df[0]
    embeddings = {}
    for word in words:
        if type(word) != str or word in embeddings:  # if the word is not nan, or it is already in dict
            continue
        embeddings[word] = torch.rand(EMBEDDING_VECTOR_SIZE)
    return embeddings


def create_random_embeddings_with_subwords(data_df):
    words = data_df[0]
    embeddings = {}
    for word in words:
        if type(word) != str:
            continue
        if len(word) > 3:
            prefix = word[:3]
            suffix = word[-3:]
            if prefix not in embeddings:
                embeddings[prefix] = torch.rand(EMBEDDING_VECTOR_SIZE)
            if suffix not in embeddings:
                embeddings[suffix] = torch.rand(EMBEDDING_VECTOR_SIZE)
        if word not in embeddings:
            embeddings[word] = torch.rand(EMBEDDING_VECTOR_SIZE)
    return embeddings


def get_accuracy(dataloader, model, tags, is_ner):
    with torch.no_grad():
        correct = 0
        samples = 0
        for x, y in dataloader:
            outputs = model(x)

            _, predictions = torch.max(outputs, 1)
            samples += y.shape[0]
            for p_i, y_i in zip(predictions, y):
                if y_i[p_i.item()] == 0:
                    continue
                if is_ner and tags[p_i.item()] == 'O':
                    samples -= 1
                    continue
                correct += 1

    return correct / samples
