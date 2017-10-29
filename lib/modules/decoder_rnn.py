import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from seq2seq.models.attention import Attention
from seq2seq.models.base_rnn import BaseRNN
from seq2seq.fields import SeqField
from seq2seq.config import configurable

logger = logging.getLogger(__name__)

# TODO: https://arxiv.org/pdf/1707.06799v1.pdf
# Consider CRF instead of softmax
# http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion

# TODO: Add an option for using the same length as the input as the max length for target
# Helps with marking


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab (torchtext.vocab.Vocab):
            an object of torchtext.vocab.Vocab class

        embedding_size (int):
            the size of the embedding for the input

        rnn_size (int):
            The number of recurrent units. Based on Nils et al., 2017, we choose the
            default value of 128. <https://arxiv.org/pdf/1707.06799v1.pdf>.

        n_layers (numpy.int, int, optional):
            Number of RNN layers used in Seq2seq. Based on Nils et
            al., 2017, we choose the default value of 2 as a "robust rule of thumb".
            <https://arxiv.org/pdf/1707.06799v1.pdf>

        rnn_cell (str, optional):
            type of RNN cell (default: gru)

        embedding_dropout (float, optional):
            dropout probability for the input sequence (default: 0)

        use_attention(bool, optional):
            Flag adds attention to the decoder. Attention is commonly used in Seq2Seq to attend to
            the encoder states. Based on wide community adoption, we recommend attention for
            Seq2Seq. <http://ruder.io/deep-learning-nlp-best-practices/index.html#fnref:27>
    """

    @configurable
    def __init__(self,
                 vocab,
                 embedding_size=100,
                 rnn_size=128,
                 n_layers=2,
                 rnn_cell='gru',
                 embedding_dropout=0,
                 rnn_dropout=0,
                 rnn_variational_dropout=0,
                 use_attention=True,
                 scheduled_sampling=False,
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

        self.eos_idx = self.vocab.stoi[SeqField.EOS_TOKEN]
        self.sos_idx = self.vocab.stoi[SeqField.SOS_TOKEN]
        self.use_attention = use_attention

        self.rnn_size = rnn_size
        self.rnn = self.rnn_cell(
            embedding_size, rnn_size, n_layers, dropout=rnn_variational_dropout)
        if use_attention:
            # TODO: Add a mask to padding index?
            self.attention = Attention(rnn_size)

        self.out = nn.Linear(rnn_size, len(self.vocab))
        self.scheduled_sampling = scheduled_sampling

    def _init_start_input(self, batch_size):
        """
        "<s>" returns a start of sequence token tensor for every sequence in the batch.

        Args:
            batch_size
        Returns:
            SOS (torch.LongTensor [1, batch_size]): Start of sequence token
        """
        init_input = Variable(
            torch.LongTensor(1, batch_size).fill_(self.sos_idx), requires_grad=False)
        if torch.cuda.is_available():
            init_input = init_input.cuda()
        return init_input

    def _get_batch_size(self, target_output, encoder_hidden):
        """
        Utility function for getting the batch_size from target_output or encoder_hidden.

        Returns:
            (int) batch size
        """
        batch_size = 1
        if target_output is not None:
            batch_size = target_output.size(1)
        else:
            if self.rnn_cell is nn.LSTM:
                batch_size = encoder_hidden[0].size(1)
            elif self.rnn_cell is nn.GRU:
                batch_size = encoder_hidden.size(1)
        return batch_size

    def forward_step(self, last_decoder_output, decoder_hidden, encoder_outputs):
        """
        Using last decoder output, decoder hidden, and encoder outputs predict the next token.

        Args:
            last_decoder_output (torch.LongTensor [output_len, batch_size]): variable containing the
                last decoder output
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
        Returns:
            predicted_softmax (torch.FloatTensor [batch_size, output_len, vocab_size]): variable containing the
                confidence for one token per sequence in the batch.
            decoder_hidden_new (tuple or tensor): variable containing the features in the hidden
                state dependant on torch.nn.GRU or torch.nn.LSTM
            attention (torch.FloatTensor [batch_size, output_len, input_len]): Attention weights on per token.
        """
        output_len = last_decoder_output.size(0)
        batch_size = last_decoder_output.size(1)
        rnn_size = self.rnn_size
        vocab_size = len(self.vocab)

        embedded = self.embedding(last_decoder_output)
        embedded = self.embedding_dropout(embedded)

        self.rnn.flatten_parameters()
        output, decoder_hidden_new = self.rnn(embedded, decoder_hidden)
        output = self.rnn_dropout(output)

        attention_weights = None
        if self.use_attention:
            # Batch first encoder_outputs
            encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()
            output = output.transpose(0, 1).contiguous()
            output, attention_weights = self.attention(output, encoder_outputs)

        # (batch_size, output_len, rnn_size) -> (batch_size * output_len, rnn_size)
        output = output.view(-1, rnn_size)
        # (batch_size * output_len, rnn_size) -> (batch_size * output_len, vocab_size)
        output = self.out(output)
        predicted_softmax = F.log_softmax(output)
        # (batch_size * output_len, vocab_size) -> (batch_size, output_len, vocab_size)
        predicted_softmax = predicted_softmax.view(batch_size, output_len, vocab_size)
        return predicted_softmax, decoder_hidden_new, attention_weights

    def _get_eos_indexes(self, decoder_output):
        """
        Args:
            decoder_output (torch.FloatTensor [batch_size, vocab_size]): decoder output for a single
                rnn pass
        Returns:
            (list) indexes of EOS tokens
        """
        predictions = decoder_output.data.topk(1)[1]
        eos_batches = predictions.view(-1).eq(self.eos_idx).nonzero()
        if eos_batches.dim() > 0:
            return eos_batches.squeeze(1).tolist()
        else:
            return []

    def forward_unrolled(self, encoder_hidden, encoder_outputs, batch_size, max_length=None):
        """
        Using encoder hidden and encoder outputs make a prediction for the decoded sequence.
        This uses the decoder output to guess the next sequence.

        Args:
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
            batch_size (int) size of the batch
        Returns:
            decoder_outputs (torch.FloatTensor [max_length - 1, batch_size, vocab_size]): outputs
                of the decoder at each timestep
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            attention_weights (torch.FloatTensor [max_length - 1, batch_size, input_len]) attention
                weights for every decoder_output 
        """
        # https://arxiv.org/pdf/1609.08144.pdf
        # https://arxiv.org/abs/1508.04025
        decoder_hidden = None if self.use_attention else encoder_hidden
        decoder_input = self._init_start_input(batch_size)
        decoder_outputs = []
        eos_tokens = set()
        attention_weights = [] if self.use_attention else None
        lengths = torch.LongTensor(batch_size).zero_()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        while True:
            decoder_output, decoder_hidden, step_attention_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)

            # (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
            decoder_output = decoder_output.squeeze(1)
            decoder_outputs.append(decoder_output)
            # (batch_size, vocab_size) -> (1, batch_size)
            decoder_input = decoder_output.max(1)[1].view(1, batch_size)
            if self.use_attention:
                # (batch_size, 1, input_len) -> (batch_size, input_len)
                step_attention_weights = step_attention_weights.squeeze(1)
                attention_weights.append(step_attention_weights)

            # Check if every batch has an eos_token
            if max_length is None:
                eos_tokens.update(self._get_eos_indexes(decoder_output))
                if len(eos_tokens) == batch_size:
                    break
                if len(decoder_outputs) == 1000:
                    logger.warn('Decoder has not predicted EOS in 1000 iterations. Breaking.')
                    break

            if max_length and len(decoder_outputs) == max_length:
                break

        decoder_outputs = torch.stack(decoder_outputs)
        if self.use_attention:
            attention_weights = torch.stack(attention_weights)

        return decoder_outputs, decoder_hidden, attention_weights

    def forward(self,
                max_length=None,
                encoder_hidden=None,
                encoder_outputs=None,
                target_output=None):
        """
        Using encoder hidden and encoder outputs make a prediction for the decoded sequence

        Args:
            max_length (int): max length of a sequence
            target_output (torch.LongTensor [max_length, batch_size]): tensor containing the target
                sequence
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
            encoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
        Returns:
            decoder_outputs (torch.FloatTensor [max_length - 1, batch_size, vocab_size]): outputs
                of the decoder at each timestep
            decoder_hidden(tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            attention_weights
        """
        if max_length is not None and max_length < 2:
            raise ValueError('Max length of 1 will only generate <s> token. Below 1, it not valid.')

        if self.use_attention and encoder_outputs is None:
            raise ValueError('Argument encoder_outputs cannot be None when attention is used.')

        batch_size = self._get_batch_size(target_output, encoder_hidden)

        if not self.scheduled_sampling and max_length and target_output is not None:
            decoder_input = torch.cat([self._init_start_input(batch_size), target_output[0:-1]])
            # https://arxiv.org/pdf/1609.08144.pdf
            # https://arxiv.org/abs/1508.04025
            decoder_hidden = None if self.use_attention else encoder_hidden
            decoder_outputs, decoder_hidden, attention_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            # (batch_size, vocab_size, output_len) -> (output_len, batch_size, vocab_size)
            decoder_outputs = decoder_outputs.transpose(0, 1).contiguous()
            if self.use_attention:
                attention_weights = attention_weights.transpose(0, 1).contiguous()
            return decoder_outputs, decoder_hidden, attention_weights

        return self.forward_unrolled(
            encoder_hidden, encoder_outputs, batch_size, max_length=max_length)
