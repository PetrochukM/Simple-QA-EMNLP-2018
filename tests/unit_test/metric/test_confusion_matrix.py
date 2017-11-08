import unittest

import numpy as np

from seq2seq.metrics import ConfusionMatrix
from tests.unit_test.metric.utils import get_batch
from tests.lib.utils import random_args


class TestConfusionMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test_init(self):
        """ Check if init fails """
        ConfusionMatrix(self.output_field)

    def test_str(self):
        option_context = ['display.max_rows', 10]
        str(ConfusionMatrix(self.output_field))
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        cm = ConfusionMatrix(self.output_field, option_context=option_context)
        cm.eval_batch(outputs, targets)
        str(cm)

    def test_same_rows_columns(self):
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        matrix = ConfusionMatrix(self.output_field, same_rows_columns=True)
        matrix.eval_batch(outputs, targets)
        data, prediction_sequences, target_sequences = matrix.get_measurement()
        self.assertEqual(set(prediction_sequences), set(target_sequences))

    def test_mask_none(self):
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = ConfusionMatrix(self.output_field, mask=None)
        metric.eval_batch(outputs, targets)
        data, index, columns = metric.get_measurement()
        self.assertTrue(np.array_equal(data, [[1, 1]]))
        self.assertEqual(len(index), 1)  # Number of prediction seq
        self.assertEqual(len(columns), 2)  # Number of target seq

    def test_mask(self):
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = ConfusionMatrix(self.output_field, mask=1)
        metric.eval_batch(outputs, targets)
        data, index, columns = metric.get_measurement()
        self.assertTrue(np.array_equal(data, [[1, 0], [0, 1]]))
        self.assertEqual(len(index), 2)  # Number of prediction seq
        self.assertEqual(len(columns), 2)  # Number of target seq

    def test_reset(self):
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = ConfusionMatrix(self.output_field, mask=1)
        metric.eval_batch(outputs, targets)
        data, index, columns = metric.get_measurement()
        self.assertTrue(np.array_equal(data, [[1, 0], [0, 1]]))
        metric.reset()
        metric.eval_batch(outputs, targets)
        data, index, columns = metric.get_measurement()
        self.assertTrue(np.array_equal(data, [[1, 0], [0, 1]]))
