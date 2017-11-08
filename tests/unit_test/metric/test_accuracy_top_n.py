import unittest

from seq2seq.metrics import AccuracyTopN
from tests.unit_test.metric.utils import get_batch


class TestAccuracyTopN(unittest.TestCase):

    def test_init(self):
        """ Check if init fails """
        AccuracyTopN(mask=None)
        AccuracyTopN(mask=1)

    def test_str(self):
        """ Check accuracy creates some string. """
        str(AccuracyTopN(mask=1))
        str(AccuracyTopN(mask=None))

    def test_mask_none_top_n_3(self):
        """ Check accuracy eval and get_measurement without mask. """
        outputs, targets = get_batch(predictions=[[2], [2]], targets=[[2], [1]])
        metric = AccuracyTopN(mask=None, top_n=3)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1)

    def test_mask_none_top_n_1(self):
        """ Check accuracy eval and get_measurement without mask. """
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = AccuracyTopN(mask=None, top_n=1)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 0.5)

    def test_mask(self):
        """ Check accuracy eval and get_measurement with mask. """
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = AccuracyTopN(mask=1, top_n=3)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1)

    def test_reset(self):
        """ Check if accuracy reset, resets the accuracy.
        """
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[2, 0]])
        metric = AccuracyTopN(mask=None, top_n=3)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1)
        metric.reset()
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[2, 3]])
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 0)
