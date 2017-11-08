import unittest
import tarfile
import os

from seq2seq.datasets.utils import get_dataset
from seq2seq.datasets.utils import urlretrieve_reporthook
from seq2seq.datasets.utils import extract_and_rename_tar_member
from seq2seq.datasets import SeqInputOutputDataset
from seq2seq.fields import SeqField
from tests.lib.utils import DATA_DIR
from tests.lib.utils import get_test_data_path


class UtilsTests(unittest.TestCase):

    def test_get_dataset(self):
        # NOTE: We do not test every dataset because many take longer than is appropriate for test
        # time.
        test_data, input_field, output_field = get_dataset(
            'ZeroToZero', DATA_DIR, load_train=True, load_dev=False, load_test=False)
        self.assertIsInstance(input_field, SeqField)
        self.assertIsInstance(output_field, SeqField)
        self.assertIsInstance(test_data, SeqInputOutputDataset)

    def test_get_dataset_share_vocab(self):
        # Make sure nothing breaks
        get_dataset(
            'ZeroToZero',
            DATA_DIR,
            load_train=True,
            load_dev=False,
            load_test=False,
            share_vocab=True)

    def test_urlretrieve_reporthook(self):
        urlretrieve_reporthook(0, 1, 100)
        urlretrieve_reporthook(1, 1, 100)

    def test_extract_and_rename_tar_member(self):
        full_path = os.path.join(get_test_data_path(), 'dev.tgz')
        new_path = os.path.join(get_test_data_path(), 'dev.tsv')
        tar = tarfile.open(full_path, 'r:gz')
        extract_and_rename_tar_member(tar, new_path, 'dev.tsv')
