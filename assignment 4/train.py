import torch
import torch.nn as nn
import utils
from bi_lstm_with_inner_attention import BiLstmWithIntraAttention


def train(train_data_iter, dev_data_iter, embedding_matrix):
    print('begin training...')
    model = BiLstmWithIntraAttention(utils.BI_LSTM_HIDDEN_DIM,
                                     utils.DROPOUT_VALUE,
                                     embedding_matrix,
                                     utils.EMBEDDING_DIM,
                                     utils.NUM_OF_BI_LSTM_LAYERS,
                                     utils.NUM_OF_CLASSES).to(utils.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=utils.LEARNING_RATE)

    for epoch in range(utils.EPOCHS):
        model.train()
        epoch_loss = 0
        counter = 0
        for batch_num, sequence_batch in enumerate(train_data_iter):
            optimizer.zero_grad()

            premise_sentences, premise_lengths = sequence_batch.premise
            hypothesis_sentences, hypothesis_lengths = sequence_batch.hypothesis

            outputs = model(premise_sentences.to(utils.device), hypothesis_sentences.to(utils.device),
                            premise_lengths, hypothesis_lengths)
            loss = criterion(outputs.to(utils.device), sequence_batch.label.to(utils.device) - 1)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss
            counter += 1

        accuracy = validate(model, dev_data_iter)
        print(f'epoch {epoch + 1}/{utils.EPOCHS}: loss - {epoch_loss / counter}, accuracy - {accuracy}')


def validate(model, dev_data_iter):
    model.eval()

    correct = 0
    samples = 0

    with torch.no_grad():
        for batch_num, sequence_batch in enumerate(dev_data_iter):
            premise_sentences, premise_lengths = sequence_batch.premise
            hypothesis_sentences, hypothesis_length = sequence_batch.hypothesis
            outputs = model(premise_sentences.to(utils.device), hypothesis_sentences.to(utils.device),
                            premise_lengths, hypothesis_length)
            _, predictions = outputs.max(1)
            correct += float((predictions == sequence_batch.label - 1).sum())
            samples += float(predictions.size(0))

    return correct / samples


def main():
    train_data_iter, dev_data_iter, test_data_iter, embedding_matrix = utils.get_data_and_embeddings()
    train(train_data_iter, dev_data_iter, embedding_matrix)


if __name__ == '__main__':
    main()
