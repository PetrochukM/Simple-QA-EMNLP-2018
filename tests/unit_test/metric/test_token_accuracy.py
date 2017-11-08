import unittest

from seq2seq.metrics import TokenAccuracy
from tests.unit_test.metric.utils import get_batch


class TestTokenAccuracy(unittest.TestCase):

    def test_init(self):
        """ Check if init fails """
        TokenAccuracy(mask=None)
        TokenAccuracy(mask=1)

    def test_mask_none(self):
        """ Check accuracy eval and get_measurement without mask. """
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [1, 1]])
        metric = TokenAccuracy(mask=None)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement()[0], .5)

    def test_mask(self):
        """ Check accuracy eval and get_measurement with mask. """
        outputs, targets = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [1, 1]])
        metric = TokenAccuracy(mask=1)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement()[0], 1)

    def test_str(self):
        """ Check accuracy creates some string. """
        # Test if __str__ runs.
        str(TokenAccuracy(mask=1))
        str(TokenAccuracy(mask=None))

    def test_reset(self):
        """ Check if accuracy reset, resets the accuracy.
        """
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[2, 0]])  # Right
        metric = TokenAccuracy(mask=None)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement()[0], 1)
        metric.reset()
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[1, 1]])  # Wrong
        metric.eval_batch(outputs, targets)
        # If reset fails, metric.get_measurement() == 0.5
        self.assertAlmostEqual(metric.get_measurement()[0], 0)
