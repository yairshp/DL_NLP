import pandas as pd

from utils import TaggerDataset
import utils
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

BEGIN_TOKEN = '-BEGIN-'
END_TOKEN = '-END-'
CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,\'"();:$-_`{}[]%#&?\\/*^@+!='
WORD_EMBEDDING_VECTOR_SIZE = 50
CHAR_EMBEDDING_VECTOR_SIZE = 20
EPOCHS = 10
HIDDEN_DIM = 1000


class CnnTaggerDataset(TaggerDataset):
    def get_embedded_data(self):
        embedded_data = []
        for sentence in self.sentences:
            for i in range(len(sentence)):
                w1 = sentence[i - 2][0] if i >= 2 else BEGIN_TOKEN
                w2 = sentence[i - 1][0] if i >= 1 else BEGIN_TOKEN
                w3 = sentence[i][0]
                w4 = sentence[i + 1][0] if i < len(sentence) - 1 else END_TOKEN
                w5 = sentence[i + 2][0] if i < len(sentence) - 2 else END_TOKEN
                x = [w1, w2, w3, w4, w5]
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


class CnnTaggerModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_dim,
                 num_of_classes,
                 characters_embedding,
                 word_embeddings):
        super(CnnTaggerModel, self).__init__()

        self.characters_embeddings = characters_embedding
        self.word_embeddings = word_embeddings

        self.conv1d = nn.Conv1d(20, 20, kernel_size=3, padding=2)
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_of_classes)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()

    def forward(self, context):
        x_temp = []
        # for context in contexts:
        context_representation = []
        for w in context:
            conv_input = self.build_conv_input_from_word(w)
            conv_input = self.dropout(conv_input)
            conv_output = self.conv1d(conv_input)
            conv_output = nn.Tanh()(conv_output)
            word_representation = torch.max(conv_output, 1)[0]
            word_embedding = self.get_word_embedding(w)
            context_representation.append(torch.concat((word_embedding, word_representation), 0))
        w1, w2, w3, w4, w5 = context_representation
        x = torch.cat((w1, w2, w3, w4, w5), 0)
        #     x_temp.append(x_i)
        # x = x_temp[0]
        # for x_i in x_temp[1:]:
        #     x = torch.cat((x, x_i), 1)
        x = self.linear1(x.float())
        x = self.tanh(x)
        x = self.linear2(x)
        return x

    def build_conv_input_from_word(self, word):
        conv_input = self.characters_embeddings[word[0][0]]
        for char in word[0][1:]:
            conv_input = torch.cat((conv_input, self.characters_embeddings[char]), 1)
        return conv_input

    def get_word_embedding(self, word):
        if word in self.word_embeddings:
            return self.word_embeddings[word]
        # elif self.are_embeddings_random:
        # else:
            # return torch.rand(WORD_EMBEDDING_VECTOR_SIZE)
        else:
            return self.word_embeddings['UUUNKKK']


def train(train_data_path, validation_data_path, chars_embeddings, w_embeddings, tags, is_ner, delimiter=" ",
          move_to_lower=False):
    train_data = CnnTaggerDataset(train_data_path, None, tags, is_train=True,
                                  delimiter=delimiter, move_to_lower=move_to_lower)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True)
    validation_data = CnnTaggerDataset(validation_data_path, None, tags, is_train=True,
                                       delimiter=delimiter, move_to_lower=move_to_lower)
    validation_dataloader = DataLoader(dataset=validation_data, shuffle=False)

    model = CnnTaggerModel(5 * (CHAR_EMBEDDING_VECTOR_SIZE + WORD_EMBEDDING_VECTOR_SIZE),
                           HIDDEN_DIM, len(tags), chars_embeddings, w_embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    accuracies = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_of_steps = 0
        for i, (x, y) in enumerate(train_dataloader):
            output = model(x)
            loss = criterion(output.view(-1, y.shape[1]), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10000 == 0:
                print(f'Epoch {epoch + 1}/{EPOCHS}, Batch {i}/{len(train_dataloader)} loss - {loss.item():.4f}')

            epoch_loss += loss
            num_of_steps += 1
        epoch_loss = epoch_loss / num_of_steps
        losses.append(epoch_loss)
        accuracy = validate(validation_dataloader, tags, model, is_ner)
        accuracies.append(accuracy)
        print(f'{epoch + 1}/{EPOCHS}: {epoch_loss}, {accuracy}')
        torch.save(model, f'saved_models/model_checkpoint_{epoch + 1}.pt')
    return model, losses, accuracies


def validate(val_dataloader, tags, model, is_ner):
    correct = 0
    samples = 0
    with torch.no_grad():
        for x, y in val_dataloader:
            samples += 1
            output = model(x)
            _, prediction = torch.max(output, 0)
            if y[0][prediction.item()] == 0:
                continue
            if is_ner and tags[prediction.item()] == 'O':
                samples -= 1
                continue
            correct += 1
    return correct / samples


def test(test_data_path, tags, model, output_file_name, delimiter=" ", move_to_lower=False):
    test_data = CnnTaggerDataset(test_data_path, None, tags, is_train=False,
                                 delimiter=delimiter, move_to_lower=move_to_lower)
    test_dataloader = DataLoader(dataset=test_data, shuffle=True)

    predictions = []
    with torch.no_grad():
        for x, y in test_dataloader:
            output = model(x)
            _, prediction = torch.max(output, 0)
            predictions.append(tags[prediction.item()])

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
    chars_embeddings = {c: torch.rand(CHAR_EMBEDDING_VECTOR_SIZE, 1) for c in CHARS}
    w_embeddings = utils.get_existing_embeddings_with_subwords(f'{utils.EMBEDDINGS_PATH}/wordVectors.txt',
                                                               f'{utils.EMBEDDINGS_PATH}/vocab.txt')
    # POS
    model, losses, accuracies = train(f'{utils.POS_DATASET_PATH}/train', f'{utils.POS_DATASET_PATH}/dev',
                                      chars_embeddings, w_embeddings, utils.POS_TAGS, is_ner=False, move_to_lower=True)
    print('losses', losses)
    print('accuracies', accuracies)
    test(f'{utils.POS_DATASET_PATH}/test', utils.POS_TAGS, model, 'test5.pos', move_to_lower=True)

    # NER
    model, losses, accuracies = train(f'{utils.NER_DATASET_PATH}/train', f'{utils.NER_DATASET_PATH}/dev',
                                      chars_embeddings, w_embeddings, utils.NER_TAGS, delimiter="\t", is_ner=True,
                                      move_to_lower=True)
    print('losses', losses)
    print('accuracies', accuracies)
    test(f'{utils.NER_DATASET_PATH}/test', utils.NER_TAGS, model, 'test5.ner', delimiter="\t", move_to_lower=True)


if __name__ == '__main__':
    main()

