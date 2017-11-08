import inspect
import random
import unittest

import torch.nn as nn

from seq2seq.metrics import Accuracy
from seq2seq.metrics import get_metrics
from seq2seq.metrics import get_loss
from tests.lib.utils import random_vocab
from tests.lib.utils import random_args
from tests.unit_test.metric.utils import MockMetric

import seq2seq.metrics


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test_get_loss(self):
        instantiation = get_loss('NLLLoss')
        self.assertTrue(isinstance(instantiation, seq2seq.metrics.Loss))

    def test_get_metrics(self):
        vocab = random_vocab()
        mask = random.randint(0, len(vocab) - 1)
        possible_kwargs = {
            'output_field': self.output_field,
            'input_field': self.input_field,
            'vocab_size': len(vocab),
            'mask': mask,
            'option_context': ['display.max_rows', 10],
            'metric': Accuracy(mask),
            'criterion': nn.NLLLoss(),
            'n_samples': random.randint(1, 5)
        }
        # Attempt to get_metrics on every class in `seq2seq.metrics`
        for name, obj in inspect.getmembers(seq2seq.metrics, inspect.isclass):
            instantiation = get_metrics([name], **possible_kwargs)[0]
            self.assertTrue(isinstance(instantiation, obj))

    def test_get_metrics_missing_args(self):
        self.assertRaises(TypeError, lambda: get_metrics(['Loss']))

    def test_get_metrics_does_not_exist(self):
        self.assertRaises(TypeError, lambda: get_metrics(['Abcabc']))

    def test_get_metrics_callable(self):
        instantiation = get_metrics([MockMetric])[0]
        self.assertTrue(isinstance(instantiation, MockMetric))
