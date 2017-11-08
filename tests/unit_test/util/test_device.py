import unittest

import torch

from seq2seq.util.device import cuda_devices
from seq2seq.util.device import device_default


class TestDevice(unittest.TestCase):

    def test_cuda_devices(self):
        """ Run CUDA devices
        """
        if torch.cuda.is_available():
            cuda_devices()
        else:
            self.assertEqual(cuda_devices(), [])

    def test_device_default(self):
        """ Check device default
        """
        self.assertEqual(device_default(1), 1)
        self.assertEqual(device_default(-1), -1)
        if torch.cuda.is_available():
            self.assertEqual(device_default(None), None)
        else:
            self.assertEqual(device_default(None), -1)
