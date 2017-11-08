import unittest

from seq2seq.datasets import Reverse
from seq2seq.datasets import SeqInputOutputDataset
from seq2seq.fields.utils import get_input_field
from seq2seq.fields.utils import get_output_field


class SeqInputOutputDatasetTest(unittest.TestCase):

    def setUp(self):
        self.input_field = get_input_field()
        self.output_field = get_output_field()

    def test_shuffle_datasets(self):
        dev_data, test_data = Reverse.splits(self.input_field, self.output_field, train=None)
        old_dev_data_examples = dev_data.examples[:]
        SeqInputOutputDataset.shuffle_datasets(dev_data, test_data)
        self.assertEqual(len(old_dev_data_examples), len(dev_data))
        self.assertTrue(any(dev_data[i] != old_dev_data_examples[i] for i in range(len(dev_data))))

    # NOTE: Every other function is well used in other tests; therefore, we do not test them.
