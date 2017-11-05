import torch.nn as nn

from lib.nn.seq_encoder import SeqEncoder
from lib.configurable import configurable


class SeqToLabel(nn.Module):

    @configurable
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 rnn_size=128,
                 rnn_cell='lstm',
                 decode_dropout=0.0):
        super().__init__()
        self.rnn_size = rnn_size
        self.rnn_cell = rnn_cell
        self.encoder = SeqEncoder(input_vocab_size, rnn_size=rnn_size, rnn_cell=rnn_cell)
        self.decode = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),  # can apply batch norm after this - add later
            nn.BatchNorm1d(self.rnn_size),
            nn.ReLU(),
            nn.Dropout(p=decode_dropout),
            nn.Linear(self.rnn_size, output_vocab_size),
            nn.Softmax())

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, text, text_lengths):
        output, hidden = self.encoder(text, text_lengths)

        if self.rnn_cell == 'gru':
            hidden = hidden
        elif self.rnn_cell == 'lstm':
            hidden = hidden[0]
        else:
            raise ValueError('Unsupported RNN_CELL')

        # [layers, batch, rnn_size] -> [batch, rnn_size]
        hidden = hidden[-1]
        # [batch, rnn_size] -> [batch, output_vocab_size]
        output = self.decode(hidden)
        return output
