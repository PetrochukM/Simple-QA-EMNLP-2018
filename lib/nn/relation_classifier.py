from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from lib.text_encoders import PADDING_INDEX
from lib.configurable import configurable
from lib.nn.lock_dropout import LockedDropout

# REFERENCE:
# https://github.com/castorini/BuboQA/blob/master/ferhan_simple_qa_rnn/relation_prediction/model.py


class Encoder(nn.Module):

    @configurable
    def __init__(self,
                 embedding_size=100,
                 rnn_size=128,
                 rnn_variational_dropout=0,
                 n_layers=2,
                 rnn_cell='gru',
                 bidirectional=True):
        super(Encoder, self).__init__()
        self.rnn_cell = rnn_cell
        if bidirectional:
            self.n_layers = n_layers * 2
        self.rnn_size = rnn_size
        self.bidirectional = bidirectional
        if self.rnn_cell.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=rnn_size,
                num_layers=n_layers,
                dropout=rnn_variational_dropout,
                bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=rnn_size,
                num_layers=n_layers,
                dropout=rnn_variational_dropout,
                bidirectional=bidirectional)

    def forward(self, inputs):
        # shape of `inputs` - (sequence length, batch size, dimension of embedding)
        batch_size = inputs.size()[1]
        state_shape = self.n_layers, batch_size, self.rnn_size
        if self.rnn_cell.lower() == "gru":
            h0 = Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(inputs, h0)
        else:
            h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(
            batch_size, -1)


class RelationClassifier(nn.Module):

    @configurable
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 freeze_embeddings=False,
                 bidirectional=False,
                 embedding_size=128,
                 embedding_dropout=0.4,
                 rnn_size=128,
                 rnn_cell='lstm',
                 decode_dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.encoder = Encoder(
            embedding_size=embedding_size,
            rnn_size=rnn_size,
            rnn_cell=rnn_cell,
            bidirectional=bidirectional)
        self.freeze_embeddings = freeze_embeddings
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if bidirectional:
            rnn_size *= 2

        self.out = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),  # can apply batch norm after this - add later
            nn.BatchNorm1d(rnn_size),
            nn.ReLU(),
            nn.Dropout(p=decode_dropout),
            nn.Linear(rnn_size, output_vocab_size))

    def forward(self, question, question_lengths):
        # shape of `batch` - (sequence length, batch size)
        question_embed = self.embedding(question)
        question_embed = self.embedding_dropout(question_embed)
        if self.freeze_embeddings:
            question_embed = Variable(question_embed.data)
        # shape of `question_embed` - (sequence length, batch size, dimension of embedding)
        question_encoded = self.encoder(question_embed)
        # shape of `question_encoded` - (batch size, number of cells X size of hidden)
        output = self.out(question_encoded)
        scores = F.log_softmax(output)
        return scores
