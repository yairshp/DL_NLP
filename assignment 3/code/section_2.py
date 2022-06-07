import random

SAME_SIZE_LEN = 200
MAX_NUM_OF_REPEATS = 20
NUM_OF_EXAMPLES = 500
A = 'a'
B = 'b'
C = 'c'
D = 'd'


def pick_num_of_repeats():
    return random.randrange(1, MAX_NUM_OF_REPEATS)


def pick_random_digit():
    return random.randrange(1, 9)


def pick_random_char():
    chars = [A, B, C, D]
    idx = random.randrange(0, 3)
    return chars[idx]


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
    for i in range(10):
        char = pick_random_char()
        sequence += generate_single_char_section(char)
        # sequence += generate_single_digit_section()
    reverse_sequence = ''
    for i in range(len(sequence) - 1, -1, -1):
        reverse_sequence += sequence[i]
    return sequence + reverse_sequence


def generate_not_palindrom():
    sequence = ''
    for i in range(10):
        char = pick_random_char()
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


def generate_with_123():
    sequence = ''
    for i in range(10):
        ch = pick_random_char()
        sequence += generate_single_char_section(ch)
        sequence += generate_single_digit_section()
    index = random.randrange(0, len(sequence) - 3)
    sequence = sequence[:index] + '123' + sequence[index:]
    return sequence


def generate_without_123():
    sequence = ''
    for i in range(10):
        ch = pick_random_char()
        sequence += generate_single_char_section(ch)
        sequence += generate_single_digit_section()
    if '123' in sequence:
        sequence.replace('123', '')
    return sequence


def write_examples_to_file(file_name, examples):
    with open(file_name, 'w') as output_file:
        for example in examples:
            output_file.write(f'{example}\n')


def main():
    # palindroms = generate_all_palindroms(500)
    # write_examples_to_file('../data/section_2/palindrom_pos_train', palindroms)
    # not_palindroms = [generate_not_palindrom() for _ in range(200)]
    # write_examples_to_file('../data/section_2/not_palindroms_test', not_palindroms)
    # odds = [generate_odds() for _ in range(200)]
    # write_examples_to_file('../data/odds_test', odds)
    # evens = [generate_evens() for _ in range(200)]
    # write_examples_to_file('../data/evens_test', evens)
    with_123 = [generate_with_123() for _ in range(200)]
    write_examples_to_file('../data/section_2/with_123_test', with_123)
    without_123 = [generate_without_123() for _ in range(200)]
    write_examples_to_file('../data/section_2/without_123_test', without_123)


if __name__ == '__main__':
    main()
