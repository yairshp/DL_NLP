import pandas as pd

POS_DEBUG_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/big_debug_50000'
POS_TRAIN_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/train'
POS_DEV_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/dev'
POS_TEST_PATH = '/home/yair/Documents/University/Deep Learning for NLP/assignment 3/data/pos/test'
UNKNOWN = 'UUUNNNKKK'
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
            'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
            'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', ',', '.', '\'\'', '(', ')', ':', '$', '#']
CHARS = ['(', ';', 'z', 'k', 'T', '5', '}', '?', '2', 'r', 'a', 'y', 'D', 's', 'G', ')', '4', '&', 'v', '8', 'K',
         'o', 'L', 'x', 'q', '`', 'm', 'g', '#', '9', '=', 'N', 'e', '1', 'P', 'I', 'd', '{', ',', '*', 'A', 'n', "'",
         'Q', 'w', 'F', 'Z', '!', 'S', 'C', 'H', 'W', '%', 'f', '/', 'c', 'i', 'E', 'U', 'M', '@', 'l', '-', '6', ':',
         'J', '7', 'R', 'u', 'Y', 'B', '.', 'p', 'j', 'V', 'O', 't', 'b', 'h', '0', 'X', '3', '$', '']


def is_end_of_sentence(word):
    return type(word) != str


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
