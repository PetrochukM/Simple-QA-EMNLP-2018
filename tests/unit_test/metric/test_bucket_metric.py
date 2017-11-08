import unittest

import numpy as np

from seq2seq.metrics import BucketMetric
from tests.unit_test.metric.utils import get_batch
from tests.lib.utils import random_args


class TestBucketMetric(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test_init(self):
        """ Check if init fails """
        BucketMetric(output_field=self.output_field, input_field=self.input_field)

    def test_str(self):
        str(BucketMetric(output_field=self.output_field, input_field=self.input_field))

    def test_bleu_metric(self):
        BucketMetric(
            output_field=self.output_field,
            input_field=self.input_field,
            metric='MosesBleu',
            mask=None)

    def test_bucket_strategy(self):
        outputs, batch = get_batch(
            sources=[[1], [1]], predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = BucketMetric(
            output_field=self.output_field,
            input_field=self.input_field,
            metric='MosesBleu',
            mask=None,
            bucket_key='source_first_token')
        metric.eval_batch(outputs, batch)
        data, keys, columns = metric.get_measurement()
        self.assertEqual(len(keys), 1)

    def test_mask_none(self):
        outputs, batch = get_batch(predictions=[[1, 0], [1, 0]], targets=[[1, 0], [1, 2]])
        metric = BucketMetric(
            output_field=self.output_field, input_field=self.input_field, mask=None)
        metric.eval_batch(outputs, batch)
        data, keys, columns = metric.get_measurement()
        self.assertTrue('0.0' in data[0])
        self.assertTrue('1.0' in data[1])
        self.assertEqual(len(keys), 2)

    def test_mask(self):
        # TODO: Allow for a larger vocab with get_batch that fits into the random_args paradim
        outputs, batch = get_batch(predictions=[[1, 2], [1, 0]], targets=[[1, 1], [1, 1]])
        metric = BucketMetric(output_field=self.output_field, input_field=self.input_field, mask=1)
        metric.eval_batch(outputs, batch)
        data, keys, columns = metric.get_measurement()
        self.assertTrue('1.0' in data[0])
        self.assertEqual(len(keys), 1)

    def test_reset(self):
        outputs, batch = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        metric = BucketMetric(
            output_field=self.output_field, input_field=self.input_field, mask=None)
        metric.eval_batch(outputs, batch)
        data, keys, columns = metric.get_measurement()
        self.assertTrue('0.5' in data[0])
        metric.reset()
        outputs, batch = get_batch(predictions=[[1, 1], [1, 1]], targets=[[1, 1], [1, 1]])
        metric.eval_batch(outputs, batch)
        data, keys, columns = metric.get_measurement()
        self.assertTrue('1.0' in data[0])
