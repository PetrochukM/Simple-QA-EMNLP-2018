import logging

import nltk

# Required for moses
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')

from nltk.tokenize.moses import MosesTokenizer

from seq2seq.util.util import get_variable_from_string

import seq2seq

logger = logging.getLogger(__name__)


def get_tokenizer(tokenizer):
    """ Get tokenizer from strings """
    if isinstance(tokenizer, _Tokenizer):
        return tokenizer
    return get_variable_from_string(tokenizer, [seq2seq.fields.utils.tokenizers])


class _Tokenizer(object):

    @classmethod
    def tokenize(cls, s):
        """ Args (str): string to tokenize """
        raise NotImplementedError()

    @classmethod
    def detokenize(cls, l):
        """ Args (list): list to detokenize """
        raise NotImplementedError()


class Moses(_Tokenizer):
    """
    Moses Tokenizer.
    """

    tokenizer = MosesTokenizer()

    @classmethod
    def tokenize(cls, s):
        return Moses.tokenizer.tokenize(s)

    @classmethod
    def detokenize(cls, l):
        return ' '.join(l)


class Word(_Tokenizer):

    @classmethod
    def tokenize(cls, s):
        return s.split()

    @classmethod
    def detokenize(cls, l):
        return ' '.join(l)


class Character(_Tokenizer):

    @classmethod
    def tokenize(cls, s):
        return list(s)

    @classmethod
    def detokenize(cls, l):
        return ''.join(l)


class Identity(_Tokenizer):

    @classmethod
    def tokenize(cls, s):
        return [s]

    @classmethod
    def detokenize(cls, l):
        return l[0]
