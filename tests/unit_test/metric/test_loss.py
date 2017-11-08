import unittest

import torch

from seq2seq.metrics.loss import Loss
from tests.lib.utils import random_args
from tests.unit_test.metric.utils import get_random_batch


class TestLoss(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test_loss_backward_with_no_loss(self):
        loss = Loss(torch.nn.NLLLoss())
        self.assertRaises(ValueError, lambda: loss.backward())


class TestNLLLoss(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)
        self.loss = Loss(torch.nn.NLLLoss())
        outputs, batch = get_random_batch(self.output_seq_len, self.input_seq_len, self.batch_size,
                                          self.output_field, self.input_field)
        self.outputs = outputs
        self.batch = batch

    def test_eval(self):
        """
        Check if NLLLoss evaluates the same as the ground truth `torch.nn.NLLLoss()`.
        """
        self.loss.eval_batch(self.outputs, self.batch)
        # Check if size_average works
        self.loss.eval_batch(self.outputs, self.batch)

        pytorch_loss = 0
        pytorch_criterion = torch.nn.NLLLoss()
        # Normalize to batch_first
        outputs = self.outputs.transpose(0, 1)
        targets = self.batch.output[0].transpose(0, 1)
        for i in range(self.batch_size):
            output = outputs[i]
            target = targets[i]
            assert output.size() == torch.Size([self.output_seq_len, len(self.output_field.vocab)])
            assert target.size() == torch.Size([self.output_seq_len])
            pytorch_loss += pytorch_criterion(output, target)

        loss_val = self.loss.get_loss()
        pytorch_loss /= self.batch_size

        self.assertAlmostEqual(loss_val, pytorch_loss.data[0], places=4)

    def test_reset(self):
        self.loss.eval_batch(self.outputs, self.batch)
        self.loss.reset()
        self.assertEqual(self.loss.get_loss(), None)

    def test_perplexity(self):
        self.loss.eval_batch(self.outputs, self.batch)
        self.assertIsInstance(self.loss.get_perplexity(), float)
