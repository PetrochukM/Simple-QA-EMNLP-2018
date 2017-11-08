import inspect
import random
import unittest

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from seq2seq.metrics import Accuracy
from seq2seq.metrics import BucketMetric
from seq2seq.metrics import ConfusionMatrix
from seq2seq.metrics import Metric
from seq2seq.metrics import get_metrics
from seq2seq.metrics import TokenAccuracy
from tests.lib.utils import random_vocab
from tests.unit_test.metric.utils import get_batch
from tests.unit_test.metric.utils import get_confidence
from tests.unit_test.metric.utils import MockMetric

import tests.unit_test.metric.test_metric
import seq2seq.metrics


class TestMetric(unittest.TestCase):

    def test_metric_init(self):
        name = "name"
        metric = Metric(name)
        self.assertEqual(metric.name, name)
