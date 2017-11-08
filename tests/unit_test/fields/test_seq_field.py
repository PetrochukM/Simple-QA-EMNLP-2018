import unittest

from seq2seq.fields import SeqField


class SeqFieldTest(unittest.TestCase):

    def setUp(self):
        # TODO: Add corpus to random args
        self.corpus = [[
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx'
        ], ["I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg']]

    def test_init(self):
        SeqField()

    def test_build_vocab(self):
        field = SeqField(tokenizer='Word')
        field.build_vocab(*self.corpus)

    def test_preprocess(self):
        field = SeqField(tokenizer='Word', lower=True, preprocessing=lambda s: s)
        self.assertTrue(isinstance(field.preprocess('This is a corpus of text'), list))

    def test_preprocess_with_vocab(self):
        field = SeqField(tokenizer='Word')
        field.build_vocab(*self.corpus)
        preprocessed = field.preprocess('This is a corpus of text')
        self.assertTrue(isinstance(field.preprocess_with_vocab(preprocessed), list))
