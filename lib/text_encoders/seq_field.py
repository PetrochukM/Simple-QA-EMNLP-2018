from collections import OrderedDict

import six

from torchtext.data import Pipeline

import torchtext

import seq2seq


class SeqField(torchtext.data.Field):
    """
    Base class for all fields used by Seq2Seq.

    Defines a sequential field.

    Wikipedia Definition of a Field: In mathematics, a field is a set on which are defined addition,
    subtraction, multiplication, and division, which behave as they do when applied to rational and
    real numbers.

    Similarly to the definition, this class defines operations on sequential inputs. The core
    operations defined are tokenize, padding and numericalize.
    """

    # NOTE: Find more documentation here:
    # https://github.com/pytorch/text/blob/master/torchtext/data/field.py

    # NOTE: For the sake of difficult spelling, we choose abbreviate 'sequential' as 'seq'.

    EOS_TOKEN = "</s>"
    SOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"

    def __init__(self,
                 lower=False,
                 tokenizer='Word',
                 preprocessing=None,
                 postprocessing=None,
                 init_token=None,
                 eos_token=None):
        tokenizer = seq2seq.fields.utils.tokenizers.get_tokenizer(tokenizer)
        tokenize = tokenizer.tokenize
        self.detokenize = tokenizer.detokenize

        super().__init__(
            sequential=True,
            use_vocab=True,
            init_token=init_token,
            eos_token=eos_token,
            fix_length=None,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            lower=lower,
            tokenize=tokenize,
            include_lengths=True,  # Required for batching in EncoderRNN#pack_padded_sequence 
            batch_first=False,  # Required for batching in EncoderRNN#pack_padded_sequence
            pad_token=SeqField.PAD_TOKEN)  # Required for batching

    def get_specials(self):
        """ Returns a list of specials ordered. """
        return list(
            OrderedDict.fromkeys(tok for tok in [self.pad_token, self.init_token, self.eos_token]
                                 if tok is not None))

    def preprocess(self, x):
        # Preprocess before vocab
        if self.sequential:
            x = self.tokenize(x)
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def preprocess_with_vocab(self, x):
        # Preprocess after `build_vocab`
        return x

    def denumeralize(self, x, ignore_eos=False):
        seq = []
        for i in x:
            seq.append(self.vocab.itos[i])
            if not ignore_eos and seq[-1] == SeqField.EOS_TOKEN:
                break
        return seq

    def decode(self, x):
        return self.detokenize(self.denumeralize(x))
