import random

MAX_NUM_OF_REPEATS = 50
NUM_OF_EXAMPLES = 500
A = 'a'
B = 'b'
C = 'c'
D = 'd'


def pick_num_of_repeats():
    return random.randrange(10, MAX_NUM_OF_REPEATS)


def pick_random_digit():
    return random.randrange(1, 9)


def generate_single_digit_section():
    section_length = pick_num_of_repeats()
    section = ''
    for i in range(section_length):
        section += str(pick_random_digit())
    return section


def generate_single_char_section(char):
    section_length = pick_num_of_repeats()
    return char * section_length


def generate_palindrom():
    sequence = ''
    for char in [A, B, C, D]:
        sequence += generate_single_char_section(char)
        # sequence += generate_single_digit_section()
    reverse_sequence = ''
    for i in range(len(sequence) - 1, -1, -1):
        reverse_sequence += sequence[i]
    return sequence + reverse_sequence


def generate_not_palindrom():
    sequence = ''
    for char in [A, B, C, D]:
        sequence += generate_single_char_section(char)
        # sequence += generate_single_digit_section()
    return sequence


def generate_all_palindroms(num_of_examples):
    return [generate_palindrom() for _ in range(num_of_examples)]


def generate_odds():
    sequence = ''
    for char in [A, B, C, D]:
        sequence += generate_single_char_section(char)
    if len(sequence) % 2 == 0:
        sequence += D
    return sequence


def generate_evens():
    sequence = ''
    for char in [A, B, C, D]:
        sequence += generate_single_char_section(char)
    if len(sequence) % 2 != 0:
        sequence += D
    return sequence


def generate_same_size():
    sequences = []
    for i in range(5, 50):
        sequence = A * i + B * i + C * i + D * i
        sequences.append(sequence)
    return sequences


def generate_different_size():
    sequence = ''
    for char in [A, B, C, D]:
        sequence += generate_single_char_section(char)
    return sequence


def write_examples_to_file(file_name, examples):
    with open(file_name, 'w') as output_file:
        for example in examples:
            output_file.write(f'{example}\n')


def main():
    # palindroms = generate_all_palindroms(200)
    # write_examples_to_file('../data/palindrom_pos_test', palindroms)
    # not_palindroms = [generate_not_palindrom() for _ in range(500)]
    # write_examples_to_file('../data/not_palindroms_train', not_palindroms)
    # odds = [generate_odds() for _ in range(200)]
    # write_examples_to_file('../data/odds_test', odds)
    # evens = [generate_evens() for _ in range(200)]
    # write_examples_to_file('../data/evens_test', evens)
    same_size = generate_same_size()
    write_examples_to_file('../data/same_size', same_size)
    different_size = [generate_different_size() for _ in range(len(same_size))]
    write_examples_to_file('../data/different_size', different_size)


if __name__ == '__main__':
    main()
