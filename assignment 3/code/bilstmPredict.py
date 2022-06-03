import argparse
from option_a import predict_model_a


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
        pass
    elif args.repr_option == 'c':
        pass
    elif args.repr_option == 'd':
        pass
    else:
        print('Error! Invalid repr option!')


if __name__ == '__main__':
    main()
