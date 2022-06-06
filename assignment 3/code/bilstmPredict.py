import argparse
from option_a import predict_model_a
from option_b import predict_model_b
from option_c import predict_model_c
from option_d import predict_model_d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('repr_option')
    parser.add_argument('model_file')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('corpus_file')
    parser.add_argument('ner_or_pos')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.repr_option == 'a':
        predict_model_a(args.model_file, args.input_file, args.output_file, args.corpus_file, args.ner_or_pos)
    elif args.repr_option == 'b':
        predict_model_b(args.model_file, args.input_file, args.output_file, args.corpus_file, args.ner_or_pos)
    elif args.repr_option == 'c':
        predict_model_c(args.model_file, args.input_file, args.output_file, args.corpus_file, args.ner_or_pos)
    elif args.repr_option == 'd':
        predict_model_d(args.model_file, args.input_file, args.output_file, args.corpus_file, args.ner_or_pos)
    else:
        print('Error! Invalid repr option!')


if __name__ == '__main__':
    main()
