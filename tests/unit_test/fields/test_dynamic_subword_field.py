import unittest

from seq2seq.fields.dynamic_subword_field import DynamicSubwordField


class DynamicSubwordFieldTest(unittest.TestCase):

    def setUp(self):
        self.corpus = [[
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx'
        ], ["I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg']]

    def test_init(self):
        DynamicSubwordField()

    def test_build_vocab(self):
        field = DynamicSubwordField()
        field.build_vocab(*self.corpus)

    def test_build_vocab_target_size(self):
        field = DynamicSubwordField()
        field.build_vocab(*self.corpus, target_size=100, min_freq=2, max_min_freq=6)

    def test_preprocess(self):
        field = DynamicSubwordField(lower=True)
        self.assertTrue(isinstance(field.preprocess('This is a corpus of text'), str))

    def test_preprocess_with_vocab(self):
        field = DynamicSubwordField(preprocessing=lambda s: s)
        field.build_vocab(*self.corpus)
        preprocessed = field.preprocess('This is a corpus of text')
        self.assertTrue(isinstance(field.preprocess_with_vocab(preprocessed), list))
