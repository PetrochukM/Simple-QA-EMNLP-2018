import unittest
import logging
import os

from lib.utils import get_root_path
from lib.utils import init_logging


class TestConfig(unittest.TestCase):

    def test_get_root_path(self):
        """ Check the root path is right.
        """
        root_path = get_root_path()
        assert os.path.isdir(root_path)
        src_dir = os.path.join(root_path, 'seq2seq')
        assert os.path.isdir(src_dir)

    def test_init_logging(self):
        """ Test if logging configuration was set.
        """
        # Remove current logging configuration
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

        # Set it up
        init_logging()

        # Something got set up
        self.assertTrue(len(logging.root.handlers) != 0)