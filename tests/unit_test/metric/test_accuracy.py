import unittest

from seq2seq.metrics import Accuracy
from tests.unit_test.metric.utils import get_batch


class TestAccuracy(unittest.TestCase):

    def test_init(self):
        """ Check if init fails """
        Accuracy(mask=None)
        Accuracy(mask=1)

    def test_str(self):
        """ Check accuracy creates some string. """
        str(Accuracy(mask=1))
        str(Accuracy(mask=None))

    def test_mask_none(self):
        """ Check accuracy eval and get_measurement without mask. """
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = Accuracy(mask=None)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), .5)

    def test_mask(self):
        """ Check accuracy eval and get_measurement with mask. """
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = Accuracy(mask=1)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1)

    def test_reset(self):
        """ Check if accuracy reset, resets the accuracy.
        """
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[2, 0]])
        metric = Accuracy(mask=None)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1)
        metric.reset()
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[2, 1]])
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 0)
