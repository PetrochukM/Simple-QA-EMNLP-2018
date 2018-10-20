import unittest
import os

import torch

from lib.utils import device_default
from lib.utils import get_root_path
from lib.utils import get_total_parameters
from lib.utils import torch_equals_ignore_index
from tests.lib.utils import random_args


class TestUtil(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test_get_root_path(self):
        """ Check the root path is right.
        """
        root_path = get_root_path()
        assert os.path.isdir(root_path)
        src_dir = os.path.join(root_path, 'lib')
        assert os.path.isdir(src_dir)

    def test_device_default(self):
        """ Check device default
        """
        self.assertEqual(device_default(1), 1)
        self.assertEqual(device_default(-1), -1)
        if torch.cuda.is_available():
            self.assertEqual(device_default(None), torch.cuda.current_device())
        else:
            self.assertEqual(device_default(None), -1)

    def test_get_total_parameters(self):
        n_params = get_total_parameters(torch.nn.LSTM(10, 10))
        self.assertTrue(n_params > 0)

    def test_torch_equals_ignore_index(self):
        self.assertTrue(
            torch_equals_ignore_index(
                torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2, 4]), ignore_index=3))
        self.assertFalse(
            torch_equals_ignore_index(
                torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2, 4]), ignore_index=4))
