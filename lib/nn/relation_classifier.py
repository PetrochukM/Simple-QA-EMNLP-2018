from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.text_encoders import PADDING_INDEX
from lib.configurable import configurable

# REFERENCE:
# https://github.com/castorini/BuboQA/blob/master/ferhan_simple_qa_rnn/relation_prediction/model.py


class RelationClassifier(nn.Module):

    @configurable
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 freeze_embeddings=False,
                 bidirectional=False,
                 embedding_size=128,
                 rnn_size=128,
                 rnn_cell='lstm',
                 rnn_layers=1,
                 decode_dropout=0.0,
                 rnn_variational_dropout=0.0):
        super().__init__()

        self.embedding = nn.Embedding(input_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.rnn_cell = rnn_cell
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_size = rnn_size
        if rnn_cell.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=rnn_size,
                num_layers=rnn_layers,
                dropout=rnn_variational_dropout,
                bidirectional=bidirectional)
        elif rnn_cell.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=rnn_size,
                num_layers=rnn_layers,
                dropout=rnn_variational_dropout,
                bidirectional=bidirectional)
        else:
            raise ValueError()

        self.decode_dropout = nn.Dropout(p=decode_dropout)
        self.relu = nn.ReLU()

        if bidirectional:
            hidden_size = rnn_size * 2

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # can apply batch norm after this - add later
            nn.BatchNorm1d(hidden_size),
            self.relu,
            self.decode_dropout,
            nn.Linear(hidden_size, output_vocab_size))

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, text, text_lengths):
        # shape of `batch` - (sequence length, batch size)
        text_embedding = self.embedding(text)
        cuda = lambda t: t.cuda() if text_embedding.is_cuda else t
        # shape of `text_embedding` - (sequence length, batch size, dimension of embedding)
        batch_size = text_embedding.size()[1]

        if self.bidirectional:
            # shape of `state_shape` - (layer * num directions, batch size, rnn size)
            state_shape = self.rnn_layers * 2, batch_size, self.rnn_size
        else:
            state_shape = self.rnn_layers, batch_size, self.rnn_size

        if self.rnn_cell.lower() == "gru":
            h0 = Variable(cuda(torch.zeros(*state_shape)))

            outputs, ht = self.rnn(text_embedding, h0)
        elif self.rnn_cell.lower() == 'lstm':
            h0 = Variable(cuda(torch.zeros(*state_shape)))
            c0 = Variable(cuda(torch.zeros(*state_shape)))

            outputs, (ht, ct) = self.rnn(text_embedding, (h0, c0))

        if self.bidirectional:
            text_encoded = ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        else:
            text_encoded = ht[-1]

        # shape of `question_encoded` - (batch size, number of cells X size of hidden)
        output = self.out(text_encoded)
        scores = F.log_softmax(output)
        return scores
