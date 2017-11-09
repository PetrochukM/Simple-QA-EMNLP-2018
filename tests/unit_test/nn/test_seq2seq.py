"""
Seq2seq unit tests.

Note: Blackbox testing of the seq2seq.
"""
import unittest

import torch

from tests.lib.utils import tensor
from tests.lib.utils import random_args
from tests.lib.utils import MockBatch


class TestSeq2Seq(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

        # Batch
        input_ = tensor(
            self.input_seq_len,
            self.batch_size,
            max_=len(self.input_field.vocab),
            type_=torch.LongTensor)
        input_lengths = torch.LongTensor(self.batch_size).fill_(self.input_seq_len)
        output = tensor(
            self.output_seq_len,
            self.batch_size,
            max_=len(self.output_field.vocab),
            type_=torch.LongTensor)
        output_lengths = torch.LongTensor(self.batch_size).fill_(self.output_seq_len - 1)
        self.batch = MockBatch(input_=(input_, input_lengths), output=(output, output_lengths))

    def test_forward(self):
        decoder_outputs, decoder_hidden, _ = self.model.forward(self.batch)

        # Check sizes
        self.assertEqual(decoder_hidden.size(), (self.num_layers, self.batch_size, self.rnn_size))
        self.assertEqual(decoder_outputs.size(), (self.output_seq_len - 1, self.batch_size,
                                                  len(self.output_field.vocab)))

        # Check types
        self.assertEqual(decoder_hidden.data.type(), 'torch.FloatTensor')
        self.assertEqual(decoder_outputs.data.type(), 'torch.FloatTensor')
