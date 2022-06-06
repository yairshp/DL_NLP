import sys
import argparse
from option_a import train_option_a
from option_b import train_option_b
from option_c import train_option_c
from option_d import train_option_d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('repr_option')
    parser.add_argument('train_file')
    parser.add_argument('model_file')
    parser.add_argument('ner_or_pos')
    parser.add_argument('dev_file')
    parser.add_argument('-v', '--vocab_file')
    parser.add_argument('-e', '--embedding_matrix_path')
    parser.add_argument('-c', '--corpus_path')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.repr_option == 'a':
        train_option_a(args.train_file, args.model_file, args.ner_or_pos, args.dev_file, args.corpus_path)
    elif args.repr_option == 'b':
        train_option_b(args.train_file, args.model_file, args.ner_or_pos, args.dev_file, args.corpus_path)
    elif args.repr_option == 'c':
        train_option_c(args.train_file, args.model_file, args.ner_or_pos, args.dev_file, args.vocab_file,
                       args.embedding_matrix_path)
    elif args.repr_option == 'd':
        train_option_d(args.train_file, args.model_file, args.ner_or_pos, args.dev_file, args.corpus_path)
    else:
        print('Error! invalid repr option!')


if __name__ == '__main__':
    main()
