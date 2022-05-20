import utils
from utils import SubWordTaggerDataset

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

LEARNING_RATE = 0.001
HIDDEN_DIM = 1000
BATCH_SIZE = 512
POS_EPOCHS = 20
NER_EPOCHS = 50


class TaggerWithSubWordsModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_of_classes):
        super(TaggerWithSubWordsModel, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, num_of_classes)

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.tanh(x)
        x = self.linear2(x)
        return x


def train(train_data_path, validation_data_path, embeddings, tags, is_ner, delimiter=" ", move_to_lower=False):
    train_data = SubWordTaggerDataset(train_data_path, embeddings, tags,
                                      is_train=True, delimiter=delimiter, move_to_lower=move_to_lower)
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

    validation_data = SubWordTaggerDataset(validation_data_path, embeddings, tags,
                                           is_train=True, delimiter=delimiter, move_to_lower=move_to_lower)
    validation_dataloader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False)

    model = TaggerWithSubWordsModel(utils.EMBEDDING_VECTOR_SIZE, HIDDEN_DIM, len(tags))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    accuracies = []
    num_of_epochs = NER_EPOCHS if is_ner else POS_EPOCHS
    for epoch in range(num_of_epochs):
        epoch_loss = 0
        num_of_steps = 0
        for i, (x, y) in enumerate(train_dataloader):
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            num_of_steps += 1
        epoch_loss = epoch_loss / num_of_steps
        losses.append(epoch_loss)
        accuracy = validate(validation_dataloader, tags, model, is_ner)
        accuracies.append(accuracy)
        print(f'{epoch + 1}/{num_of_epochs}: {epoch_loss}, {accuracy}')
    return model, losses, accuracies


def validate(validation_dataloader, tags, model, is_ner):
    return utils.get_accuracy(validation_dataloader, model, tags, is_ner)


def test(test_data_path, embeddings, tags, model, output_file_name, delimiter=" ", move_to_lower=False):
    test_data = SubWordTaggerDataset(test_data_path, embeddings, tags,
                                     is_train=False, delimiter=delimiter, move_to_lower=move_to_lower)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for x, y in test_dataloader:
            outputs = model(x)
            _, batch_predictions = torch.max(outputs, 1)
            for p_i in batch_predictions:
                predictions.append(tags[p_i.item()])

    print('writing output file')
    num_of_words = 0
    test_df = pd.read_csv(test_data_path, delimiter=delimiter, header=None, skip_blank_lines=False, quoting=3)
    with open(output_file_name, 'w') as writer:
        for w in test_df[0]:
            if type(w) != str:
                writer.write('\n')
                continue
            writer.write(f'{w} {predictions[num_of_words]}\n')
            num_of_words += 1


def main():
    # Existing Embeddings
    embeddings = utils.get_existing_embeddings_with_subwords(f'{utils.EMBEDDINGS_PATH}/wordVectors.txt',
                                                             f'{utils.EMBEDDINGS_PATH}/vocab.txt')

    # POS
    model, losses, accuracies = train(f'{utils.POS_DATASET_PATH}/train', f'{utils.POS_DATASET_PATH}/dev',
                                      embeddings, utils.POS_TAGS, is_ner=False, move_to_lower=True)
    print('losses', losses)
    print('accuracies', accuracies)
    test(f'{utils.POS_DATASET_PATH}/test', embeddings, utils.POS_TAGS, model, 'test4.pos', move_to_lower=True)

    # NER
    model, losses, accuracies = train(f'{utils.NER_DATASET_PATH}/train', f'{utils.NER_DATASET_PATH}/dev',
                                      embeddings, utils.NER_TAGS, is_ner=True, delimiter="\t", move_to_lower=True)
    print('losses', losses)
    print('accuracies', accuracies)
    test(f'{utils.NER_DATASET_PATH}/test', embeddings, utils.NER_TAGS, model, 'test4.ner',
         delimiter="\t", move_to_lower=True)

    # Random Embeddings

    # POS
    pos_df = pd.read_csv(f'{utils.POS_DATASET_PATH}/train', delimiter=" ", header=None, skip_blank_lines=False)
    embeddings = utils.create_random_embeddings(pos_df)
    model, losses, accuracies = train(f'{utils.POS_DATASET_PATH}/train', f'{utils.POS_DATASET_PATH}/dev',
                                      embeddings, utils.POS_TAGS, is_ner=False)
    print('losses', losses)
    print('accuracies', accuracies)

    # NER
    ner_df = pd.read_csv(f'{utils.NER_DATASET_PATH}/train', header=None, delimiter="\t", skip_blank_lines=False)
    embeddings = utils.create_random_embeddings(ner_df)
    model, losses, accuracies = train(f'{utils.NER_DATASET_PATH}/train', f'{utils.NER_DATASET_PATH}/dev',
                                      embeddings, utils.NER_TAGS, is_ner=True, delimiter="\t")
    print('losses', losses)
    print('accuracies', accuracies)


if __name__ == '__main__':
    main()
