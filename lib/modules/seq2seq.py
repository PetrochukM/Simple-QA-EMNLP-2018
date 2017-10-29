import torch
import torch.nn as nn

from seq2seq.config import configurable


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

    def forward(self, batch):
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
        input_, lengths = batch.input
        encoder_outputs, encoder_hidden = self.encoder(input_, lengths)

        max_length = None
        output = None
        if self.training and hasattr(batch, 'output'):
            output = batch.output[0]
        # Use max_length to cutoff prediction during evaluation.
        if hasattr(batch, 'output'):
            output_lengths = batch.output[1]
            max_length = torch.max(output_lengths)

        return self.decoder(
            max_length=max_length,
            target_output=output,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs)
