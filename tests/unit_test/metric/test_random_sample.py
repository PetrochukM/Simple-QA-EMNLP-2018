import unittest

from seq2seq.metrics import RandomSample
from tests.lib.utils import random_args
from tests.unit_test.metric.utils import get_random_batch


class TestRandomSample(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)
        outputs, batch = get_random_batch(self.output_seq_len, self.input_seq_len, self.batch_size,
                                          self.output_field, self.input_field)
        self.outputs = outputs
        self.batch = batch

    def test_init(self):
        """ Check if init fails """
        RandomSample(output_field=self.output_field, input_field=self.input_field, n_samples=3)

    def test_get_measurement(self):
        metric = RandomSample(
            output_field=self.output_field, input_field=self.input_field, n_samples=self.batch_size)
        metric.eval_batch(self.outputs, self.batch)
        metric.eval_batch(self.outputs, self.batch)
        metric.eval_batch(self.outputs, self.batch)
        self.assertEqual(
            max(len(metric.get_measurement()[0]), len(metric.get_measurement()[1])),
            self.batch_size)

    def test_str(self):
        """ Check random sample creates some string. """
        self.assertIsInstance(
            str(
                RandomSample(
                    output_field=self.output_field, input_field=self.input_field, n_samples=3)),
            str)

    def test_reset(self):
        metric = RandomSample(
            output_field=self.output_field, input_field=self.input_field, n_samples=1)
        metric.eval_batch(self.outputs, self.batch)
        output = str(metric)
        metric.reset()
        other_output = str(metric)
        self.assertNotEqual(output, other_output)
