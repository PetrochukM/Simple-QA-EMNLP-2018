import torch
import torch.nn as nn

from seq2seq.models.base_rnn import BaseRNN
from seq2seq.config import configurable


class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab (torchtext.vocab.Vocab):
            an object of torchtext.vocab.Vocab class

        embedding_size (int):
            the size of the embedding for the input

        rnn_size (int):
            The number of recurrent units. Based on Nils et al., 2017, we choose the
            default value of 128. <https://arxiv.org/pdf/1707.06799v1.pdf>. 

        embedding_dropout (float, optional):
            dropout probability for the input sequence

        n_layers (numpy.int, int, optional):
            Number of RNN layers used in Seq2seq. Based on Nils et
            al., 2017, we choose the default value of 2 as a "robust rule of thumb".
            <https://arxiv.org/pdf/1707.06799v1.pdf>

        rnn_cell (str, optional):
            type of RNN cell

        bidirectional (bool, optional):
            Flag adds directionality to the encoder. Bidirectional encoders outperform
            unidirectional ones by a small margin.
            <http://ruder.io/deep-learning-nlp-best-practices/index.html#fnref:27>
    """

    @configurable
    def __init__(self,
                 vocab,
                 embedding_size=100,
                 rnn_size=128,
                 embedding_dropout=0,
                 rnn_dropout=0,
                 rnn_variational_dropout=0,
                 n_layers=2,
                 rnn_cell='gru',
                 bidirectional=True,
                 freeze_embeddings=False):
        super().__init__(
            vocab=vocab,
            embedding_size=embedding_size,
            embedding_dropout=embedding_dropout,
            rnn_dropout=rnn_dropout,
            rnn_cell=rnn_cell,
            freeze_embeddings=freeze_embeddings)
        n_layers = int(n_layers)
        rnn_size = int(rnn_size)

        # NOTE: This assert is included because PyTorch throws a weird error if layers==0
        assert n_layers > 0, """There must be more than 0 layers."""

        # Bidirectional doubles the RNN size per direction
        if bidirectional:
            assert rnn_size % 2 == 0, """RNN size must be divisible by two. This ensures
              consistency between the Bidirectional Encoder RNN hidden state size and the Decoder
              RNN hidden state size."""
            rnn_size = rnn_size // 2

        self.bidirectional = bidirectional
        self.rnn = self.rnn_cell(
            embedding_size,
            rnn_size,
            n_layers,
            bidirectional=bidirectional,
            dropout=rnn_variational_dropout)

    def forward(self, input_, lengths):
        """
        Args:
            input_: (torch.LongTensor [seq_len, batch_size]): variable containing the encoded
                features of the input sequence
            lengths: (torch.LongTensor [batch_size]): tensor containing the lengths of each input_
                sequence
        Returns:
            outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing the
                encoded features of the input sequence
            hidden (tuple or tensor): variable containing the features in the hidden state dependant
                on torch.nn.GRU or torch.nn.LSTM
        """
        embedded = self.embedding(input_)
        embedded = self.embedding_dropout(embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.tolist())
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = self.rnn_dropout(output)

        # Encoder has one or two hidden states depending on LSTM or GRU
        if isinstance(hidden, tuple):
            hidden = tuple(self._reshape_encoder_hidden(h) for h in hidden)
        else:
            hidden = self._reshape_encoder_hidden(hidden)

        return output, hidden

    def _reshape_encoder_hidden(self, hidden):
        """
        Without Bidirectional RNN, the size of the hidden state is (layers, batch, directions *
        rnn_size).
        
        Bidirectional has a different shape (layers * directions, batch, rnn_size / 2). This
        function helps normalize the output.

        Args:
            hidden (torch.Tensor [layers * directions, batch, rnn_size / 2]): encoder hidden state
        Returns:
            hidden (torch.Tensor [layers, batch, directions * rnn_size]): reshaped hidden state
        """
        if self.bidirectional:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)

        return hidden
