import unittest
import inspect

from seq2seq.fields.utils import get_tokenizer
from seq2seq.fields.utils.tokenizers import Character
from seq2seq.fields.utils.tokenizers import Word
from seq2seq.fields.utils.tokenizers import _Tokenizer

import seq2seq


class TestTokenizers(unittest.TestCase):

    def test_get_tokenizers(self):
        # Attempt to get_metrics on every class in `seq2seq.metrics`
        for name, obj in inspect.getmembers(seq2seq.fields.utils.tokenizers, inspect.isclass):
            if name != '_Tokenizer' and issubclass(obj, _Tokenizer):
                instantiation = get_tokenizer(name)
                source = 'hi there'
                self.assertTrue(
                    instantiation.detokenize(instantiation.tokenize('hi there')), 'hi there')

    def test_character(self):
        self.assertEqual(Character.tokenize('hi there'), ['h', 'i', ' ', 't', 'h', 'e', 'r', 'e'])

    def test_word(self):
        self.assertEqual(Word.tokenize('hi there'), ['hi', 'there'])
