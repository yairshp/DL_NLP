import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import utils


class BiLstmWithIntraAttention(nn.Module):
    def __init__(self, bi_lstm_hidden_dim, dropout_value, embeddings_matrix, embedding_dim, num_of_bi_lstm_layers,
                 num_of_classes):
        super(BiLstmWithIntraAttention, self).__init__()

        self.bi_lstm_hidden_dim = bi_lstm_hidden_dim
        self.dropout = nn.Dropout(dropout_value)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings=embeddings_matrix, freeze=True)
        self.bi_lstm = nn.LSTM(input_size=embedding_dim,
                               hidden_size=self.bi_lstm_hidden_dim,
                               num_layers=num_of_bi_lstm_layers,
                               batch_first=True,
                               dropout=dropout_value,
                               bidirectional=True)

        self.w_y = nn.Linear(2 * bi_lstm_hidden_dim, 2 * bi_lstm_hidden_dim)
        self.w_h = nn.Linear(2 * bi_lstm_hidden_dim, 2 * bi_lstm_hidden_dim)
        self.w = nn.Linear(2 * bi_lstm_hidden_dim, 1)
        self.w_y.weight.data = nn.init.xavier_normal_(self.w_y.weight.data)
        self.w_h.weight.data = nn.init.xavier_normal_(self.w_h.weight.data)
        self.w.weight.data = nn.init.xavier_normal_(self.w.weight.data)

        self.linear = nn.Linear(8 * bi_lstm_hidden_dim, num_of_classes)

        self.linear_1 = nn.Linear(8 * bi_lstm_hidden_dim, embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_3 = nn.Linear(embedding_dim, num_of_classes)
        self.linear_1.weight.data = nn.init.xavier_normal_(self.linear_1.weight.data)
        self.linear_2.weight.data = nn.init.xavier_normal_(self.linear_2.weight.data)
        self.linear_3.weight.data = nn.init.xavier_normal_(self.linear_3.weight.data)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, premises, hypothesis, premises_lengths, hypothesis_lengths):
        premise_bi_lstm_output = self.apply_bi_lstm(premises, premises_lengths)
        premise_mean_pooling_output = torch.mean(premise_bi_lstm_output, dim=1, keepdim=True).permute(0, 2, 1)
        premise_inner_attention_output = self.apply_inner_attention(premise_bi_lstm_output, premise_mean_pooling_output)

        hypothesis_bi_lstm_output = self.apply_bi_lstm(hypothesis, hypothesis_lengths)
        hypothesis_mean_pooling_output = torch.mean(hypothesis_bi_lstm_output, dim=1, keepdim=True).permute(0, 2, 1)
        hypothesis_inner_attention_output = self.apply_inner_attention(hypothesis_bi_lstm_output,
                                                                       hypothesis_mean_pooling_output)

        difference = premise_inner_attention_output - hypothesis_inner_attention_output
        multiplication = premise_inner_attention_output * hypothesis_inner_attention_output
        sentence_matching_output = torch.cat((premise_inner_attention_output, difference, multiplication,
                                              hypothesis_inner_attention_output), dim=1)

        return self.linear(sentence_matching_output)

    def apply_bi_lstm(self, sentences, length):
        embedded_sentences = self.embedding_layer(sentences)
        embedded_sentences = self.dropout(embedded_sentences)
        packed_sentences = pack_padded_sequence(embedded_sentences, length.cpu(),
                                                enforce_sorted=False, batch_first=True)
        out, (_, _) = self.bi_lstm(packed_sentences)
        out, _ = pad_packed_sequence(out, batch_first=True)

        return out

    def apply_inner_attention(self, bi_lstm_output, mean_pooling_output):
        w_y = self.w_y(bi_lstm_output)
        r_ave_multiply_e_l = torch.matmul(mean_pooling_output, torch.ones((1, bi_lstm_output.shape[1]),
                                                                          device=utils.device))
        w_h = self.w_h(r_ave_multiply_e_l.permute(0, 2, 1))
        m = self.tanh(w_y + w_h)
        wm = self.w(m).squeeze(2)
        alpha = self.softmax(wm)
        r_att = torch.bmm(m.permute(0, 2, 1), alpha.unsqueeze(2))
        r_att = r_att.squeeze(2)
        return r_att

