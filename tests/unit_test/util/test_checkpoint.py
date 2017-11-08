import mock
import os
import shutil
import unittest

from seq2seq.util.checkpoint import Checkpoint
from tests.lib.utils import random_args


class TestCheckpoint(unittest.TestCase):

    def tearDown(self):
        if os.path.exists(self.experiments_directory):
            shutil.rmtree(self.experiments_directory)

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args().items():
            setattr(self, key, value)

        self.optimizer_state_dict = self.optimizer.state_dict()

    def test_init(self):
        Checkpoint(self.model, self.optimizer_state_dict, self.input_field, self.output_field)

    @mock.patch('seq2seq.util.checkpoint.os.listdir')
    def test_get_latest_checkpoint_name(self, mock_listdir):
        """ Check the right latest checkpoint name is picked.
        """
        mock_listdir.return_value = [
            '2017_05_22_09_47_26', '2017_05_22_09_47_31', '2017_05_23_10_47_29'
        ]
        latest_checkpoint = Checkpoint.get_latest_checkpoint_name(self.experiments_directory)
        self.assertTrue('2017_05_23_10_47_29' in latest_checkpoint)

    def test_save_load_checkpoint(self):
        """ Try to save and load a checkpoint.
        """
        checkpoint = Checkpoint(
            model=self.model,
            optimizer_state_dict=self.optimizer_state_dict,
            input_field=self.input_field,
            output_field=self.output_field)
        checkpoint.save(self.experiments_directory)
        latest_checkpoint = Checkpoint.get_checkpoint(self.experiments_directory)
        # Check the same attributes are set
        self.assertEqual(dir(latest_checkpoint), dir(checkpoint))
