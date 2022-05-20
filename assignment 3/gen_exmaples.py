import random

MAX_NUM_OF_REPEATS = 50
NUM_OF_EXAMPLES = 500
A = 'a'
B = 'b'
C = 'c'
D = 'd'


def pick_num_of_repeats():
    return random.randrange(1, MAX_NUM_OF_REPEATS)


def pick_random_digit():
    return random.randrange(0, 9)


def generate_single_digit_section():
    section_length = pick_num_of_repeats()
    section = ''
    for i in range(section_length):
        section += str(pick_random_digit())
    return section


def generate_single_char_section(char):
    section_length = pick_num_of_repeats()
    return char * section_length


def generate_single_sequence(is_pos):
    sequence = generate_single_digit_section()
    chars = [A, B, C, D] if is_pos else [A, C, B, D]
    for char in chars:
        sequence += generate_single_char_section(char)
        sequence += generate_single_digit_section()
    return sequence


def generate_sequences(is_pos, num_of_sequences):
    sequences = set()
    for i in range(num_of_sequences):
        sequences.add(generate_single_sequence(is_pos))
    return sequences


def main():
    positive_sequences = generate_sequences(True, NUM_OF_EXAMPLES)
    negative_sequences = generate_sequences(False, NUM_OF_EXAMPLES)
    pass


if __name__ == '__main__':
    main()
