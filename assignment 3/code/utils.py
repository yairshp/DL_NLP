import numpy as np
import pandas as pd
import torch

POS_DEBUG_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/big_debug_50000'
POS_TRAIN_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/train'
POS_DEV_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/dev'
POS_TEST_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/test'
NER_DEBUG_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/ner/debug'
NER_TRAIN_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/ner/train'
NER_DEV_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/ner/dev'
NER_TEST_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/ner/test'
EMBEDDING_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/embeddings'
UNKNOWN = 'UUUNKKK'
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
            'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
            'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', ',', '.', '\'\'', '(', ')', ':', '$', '#']
# NER_TAGS = ['PER', 'LOC', 'ORG', 'TIME', 'O', 'MISC']
NER_TAGS = ['PER', 'LOC', 'ORG', 'O', 'MISC']
CHARS = ['(', ';', 'z', 'k', 'T', '5', '}', '?', '2', 'r', 'a', 'y', 'D', 's', 'G', ')', '4', '&', 'v', '8', 'K', '"',
         'o', 'L', 'x', 'q', '`', 'm', 'g', '#', '9', '=', 'N', 'e', '1', 'P', 'I', 'd', '{', ',', '*', 'A', 'n', "'",
         'Q', 'w', 'F', 'Z', '!', 'S', 'C', 'H', 'W', '%', 'f', '/', 'c', 'i', 'E', 'U', 'M', '@', 'l', '-', '6', ':',
         'J', '7', 'R', 'u', 'Y', 'B', '.', 'p', 'j', 'V', 'O', 't', 'b', 'h', '0', 'X', '3', '$', '', '+', '[', ']']
PREFIX_LEN = 3
SUFFIX_LEN = 3


def is_end_of_sentence(word):
    return type(word) != str


def create_corpus(corpus_path, delimiter=' '):
    raw_data = pd.read_csv(corpus_path, delimiter=delimiter, header=None, quoting=3)
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


def get_suffix(word):
    if len(word) <= SUFFIX_LEN:
        return word
    return word[-SUFFIX_LEN:]


def get_prefix(word):
    if len(word) <= PREFIX_LEN:
        return word
    return word[:PREFIX_LEN]


def create_corpus_with_subwords(vocab_path):
    with open(vocab_path, 'r') as r:
        vocab = r.readlines()
    words = [w[:-1] for w in vocab]
    words_counter = 0
    prefix_counter = 0
    suffix_counter = 0
    corpus = {}
    prefixes = {}
    suffixes = {}
    for word in words:
        if word not in corpus:
            corpus[word] = words_counter
            words_counter += 1
        prefix = get_prefix(word)
        if prefix not in prefixes:
            prefixes[prefix] = prefix_counter
            prefix_counter += 1
        suffix = get_suffix(word)
        if suffix not in suffixes:
            suffixes[suffix] = suffix_counter
            suffix_counter += 1
    # corpus[UNKNOWN] = words_counter
    return corpus, prefixes, suffixes


def create_embedding_matrix(vectors_path):
    vectors = np.loadtxt(vectors_path)
    return torch.stack([torch.tensor(v) for v in vectors])
