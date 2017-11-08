# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import random

import numpy as np

from seq2seq.metrics import ApproxBleu
from seq2seq.metrics import MosesBleu
from seq2seq.metrics.bleu import bleu_score
from seq2seq.metrics.bleu import moses_multi_bleu
from tests.lib.utils import random_args
from tests.unit_test.metric.utils import get_batch


class TestBleuScore(unittest.TestCase):

    def testComputeBleuEqual(self):
        translation_corpus = [[1, 2, 3]]
        reference_corpus = [[1, 2, 3]]
        bleu = bleu_score(reference_corpus, translation_corpus)
        actual_bleu = 1.0
        self.assertEqual(bleu, actual_bleu)

    def testComputeNotEqual(self):
        translation_corpus = [[1, 2, 3, 4]]
        reference_corpus = [[5, 6, 7, 8]]
        bleu = bleu_score(reference_corpus, translation_corpus)
        actual_bleu = 0.0
        self.assertEqual(bleu, actual_bleu)

    def testComputeMultipleBatch(self):
        translation_corpus = [[1, 2, 3, 4], [5, 6, 7, 0]]
        reference_corpus = [[1, 2, 3, 4], [5, 6, 7, 10]]
        bleu = bleu_score(reference_corpus, translation_corpus)
        actual_bleu = 0.7231
        self.assertAlmostEqual(bleu, actual_bleu, delta=.0001)

    def testComputeMultipleNgrams(self):
        reference_corpus = [[1, 2, 1, 13], [12, 6, 7, 4, 8, 9, 10]]
        translation_corpus = [[1, 2, 1, 3], [5, 6, 7, 4]]
        bleu = bleu_score(reference_corpus, translation_corpus)
        actual_bleu = 0.486
        self.assertAlmostEqual(bleu, actual_bleu, delta=.0001)


class TestMosesBleu(unittest.TestCase):
    """Tests using the Moses multi-bleu script to calculate BLEU score"""

    def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
        """Runs a multi-bleu test."""
        result = moses_multi_bleu(hypotheses=hypotheses, references=references, lowercase=lowercase)
        np.testing.assert_almost_equal(result, expected_bleu, decimal=2)

    def test_multi_bleu(self):
        self._test_multi_bleu(
            hypotheses=np.array(
                ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"]),
            references=np.array([
                "The quick brown fox jumps over the lazy dog 笑",
                "The quick brown fox jumps over the lazy dog 笑"
            ]),
            lowercase=False,
            expected_bleu=46.51)

    def test_empty(self):
        self._test_multi_bleu(
            hypotheses=np.array([]), references=np.array([]), lowercase=False, expected_bleu=0.00)

    def test_multi_bleu_lowercase(self):
        self._test_multi_bleu(
            hypotheses=np.array(
                ["The brown fox jumps over The Dog 笑", "The brown fox jumps over The Dog 2 笑"]),
            references=np.array([
                "The quick brown fox jumps over the lazy dog 笑",
                "The quick brown fox jumps over the lazy dog 笑"
            ]),
            lowercase=True,
            expected_bleu=46.51)


class TestApproxBleuMetric(unittest.TestCase):

    def test_init(self):
        ApproxBleu(mask=None)
        ApproxBleu(mask=1)

    def test_str(self):
        str(ApproxBleu(mask=1))
        str(ApproxBleu(mask=None))

    def test_mask_none(self):
        outputs, targets = get_batch(
            predictions=[[1, 2, 3, 4], [5, 6, 7, 0]],
            targets=[[1, 2, 3, 4], [5, 6, 7, 10]],
            vocab_size=11)
        metric = ApproxBleu(mask=None)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 0.7231, places=3)

    def test_mask(self):
        outputs, targets = get_batch(
            predictions=[[1, 2, 3, 4], [5, 6, 7, 10]],
            targets=[[1, 2, 1, 1], [5, 6, 7, 1]],
            vocab_size=11)
        metric = ApproxBleu(mask=1)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1.0)

    def test_reset(self):
        """ Check if accuracy reset, resets the accuracy.
        """
        outputs, targets = get_batch(predictions=[[2, 0]], targets=[[2, 0]], vocab_size=11)
        metric = ApproxBleu(mask=None)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 1)
        metric.reset()
        outputs, targets = get_batch(predictions=[[1, 2]], targets=[[3, 4]], vocab_size=11)
        metric.eval_batch(outputs, targets)
        self.assertAlmostEqual(metric.get_measurement(), 0.0)


class TestMosesBleuMetric(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test(self):
        random_target = [
            random.randint(0, len(self.output_field.vocab) - 1) for _ in range(self.output_seq_len)
        ]
        outputs, targets = get_batch(
            predictions=[random_target],
            targets=[random_target],
            vocab_size=len(self.output_field.vocab))
        metric = MosesBleu(output_field=self.output_field, mask=1)
        metric.eval_batch(outputs, targets)
        metric.get_measurement()
        self.assertTrue(metric.get_measurement() >= 0.0)
        self.assertTrue(metric.get_measurement() <= 100.0)
