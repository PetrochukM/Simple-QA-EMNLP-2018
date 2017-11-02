import torch
import torch.nn as nn

from lib.configurable import configurable


class SeqToLabel(nn.Module):

    @configurable
    def __init__(self, encoder, output_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(self.encoder.rnn_size, output_vocab_size)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, source, source_lengths, target=None):
        output, hidden = self.encoder(source, source_lengths)
        # output torch.FloatTensor [seq_len, batch_size, rnn_size]
        output = output[output.size()[0] - 1]  # Grab the last hidden state
        output = output.view(-1, output.size(2))  # [seq_len * batch_size, rnn_size]
        decoded = self.decoder()
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return result, hidden
