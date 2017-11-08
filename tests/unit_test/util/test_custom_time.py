import unittest

from seq2seq.util.custom_time import pretty_time


class TestCustomTime(unittest.TestCase):

    def test_pretty_time(self):
        # Example from usage in module comment.
        self.assertEqual(pretty_time(426753), '4d 22h 32m 33s')
