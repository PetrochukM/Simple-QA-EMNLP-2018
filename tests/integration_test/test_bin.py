import unittest
import yaml
import os

from seq2seq.config import get_root_path
from seq2seq.controllers.predictor import Predictor
from seq2seq.util.checkpoint import Checkpoint

from bin.evaluate import main as evaluate
from bin.train import main as train

# TODO: Add a test for tune


class BinTest(unittest.TestCase):
    # NOTE: Numbers are used to order tests

    def setUp(self):
        self.config_file = 'configs/zero_to_zero.yml'
        self.tune_file = 'configs/zero_to_zero_tune.yml'
        self.config = yaml.load(open(os.path.join(get_root_path(), self.config_file)))

    def test_0_train(self):
        train(self.config)

    def test_1_train_checkpoint(self):
        checkpoint_name = Checkpoint.get_latest_checkpoint_name()
        train(self.config, checkpoint_name)

    def test_1_evaluate(self):
        evaluate(self.config)

    def test_1_predict(self):
        checkpoint = Checkpoint.get_checkpoint()
        predictor = Predictor(checkpoint.model, checkpoint.input_field, checkpoint.output_field)
        output_sequence = predictor.predict('0')[0]
        self.assertIsInstance(output_sequence, list)
