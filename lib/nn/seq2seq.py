import torch
import torch.nn as nn

from lib.configurable import configurable


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    """

    @configurable
    def __init__(self, encoder, decoder, tie_weights=False):
        """
        Args:
            encoder (EncoderRNN): object of EncoderRNN
            decoder (DecoderRNN): object of DecoderRNN
        """
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if tie_weights:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, source, source_lengths, target=None, target_lengths=None):
        """
        Args:
            batch (torchtext.data.Batch): Batch object with an input and output field containing the
                encoded features of the input sequence and output sequence
        Returns:
            decoder_outputs (torch.FloatTensor [max_length - 1, batch_size, rnn_size]): outputs
                of the decoder at each timestep
            decoder_hidden(tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            ret_dict (dict): dictionary containing additional information as follows: {
                *KEY_LENGTH* : list of integers representing lengths of output sequences,
                *KEY_SEQUENCE* : list of sequences, where each sequence is a list of predicted token
                    IDs,
                *KEY_INPUT* : target outputs if provided for decoding,
                *KEY_ATTN_SCORE* : list of sequences, where each list is of attention weights}
        """
        encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)

        if target is not None:
            max_length = torch.max(target_lengths)
        else:
            max_length = None

        return self.decoder(
            max_length=max_length,
            target_output=target,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs)
