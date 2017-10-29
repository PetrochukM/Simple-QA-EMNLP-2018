from collections import Counter

import torchtext

from seq2seq.fields.utils import SubwordTextTokenizer
from seq2seq.fields.seq_field import SeqField


# NOTE: With Dynamic, we mean that the Field preprocessing is determined based off of data rather
# than having a static preprocessing routine.
class DynamicSubwordField(SeqField):
    """
    Missing function definition can be found here:
    https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py

    Subword tokenization depends on the vocab; therefore, we remove tokenization during
    preprocessing and add it to numericalize.
    """

    def __init__(self,
                 lower=False,
                 preprocessing=None,
                 postprocessing=None,
                 init_token=None,
                 eos_token=None):
        super().__init__(
            lower=lower,
            init_token=init_token,
            eos_token=eos_token,
            preprocessing=preprocessing,
            postprocessing=postprocessing)

    def preprocess(self, x):
        # Preprocess before `build_vocab`
        if self.lower:
            x = x.lower()
        return x

    def preprocess_with_vocab(self, x):
        # Preprocess after `build_vocab`
        if self.sequential:
            x = self.tokenize(x)
        if self.preprocessing is not None:
            return self.preprocessing(x)
        return x

    def build_vocab(self, *args, target_size=None, min_freq=1, max_min_freq=1e3, **kwargs):
        # Build up sources
        sources = []
        for arg in args:
            if isinstance(arg, torchtext.data.Dataset):
                sources += [
                    getattr(arg, name) for name, field in arg.fields.items() if field is self
                ]
            else:
                sources.append(arg)

        # Learn a subword vocab
        if target_size is None:
            tokenizer = SubwordTextTokenizer()
            tokenizer.build_from_corpus(*sources, min_count=min_freq)
        else:
            tokenizer = SubwordTextTokenizer.build_to_target_size_from_corpus(
                *sources, target_size=target_size, min_val=min_freq, max_val=max_min_freq)

        # NOTE: Remove this field when it is supported in Pytorch/text
        # https://github.com/pytorch/text/issues/82
        counter = Counter()
        counter.update(tokenizer.vocab())
        self.tokenize = tokenizer.encode
        self.detokenize = tokenizer.decode
        self.vocab = torchtext.vocab.Vocab(counter, specials=self.get_specials(), **kwargs)
        return self
