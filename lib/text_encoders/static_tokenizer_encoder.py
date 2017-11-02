import torch

from lib.text_encoders.reserved_tokens import EOS_INDEX
from lib.text_encoders.reserved_tokens import UNKNOWN_INDEX
from lib.text_encoders.reserved_tokens import RESERVED_ITOS
from lib.text_encoders.reserved_tokens import RESERVED_STOI
from lib.text_encoders.text_encoders import TextEncoder


class StaticTokenizerEncoder(TextEncoder):
    """ Encoder where the tokenizer is not learned and a static function. """

    def __init__(self, sample, append_eos=True, lower=True, tokenize=(lambda s: s.split())):
        """ Given a sample, build the dictionary for the word encoder """
        self.lower = lower
        self.tokenize = tokenize
        self.append_eos = append_eos
        self.vocab = set()

        for text in sample:
            self.vocab.update(self._preprocess(text))

        self.stoi = RESERVED_STOI.copy()
        self.itos = RESERVED_ITOS[:]
        for token in self.vocab:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    @property
    def vocab_size(self):
        return len(self.itos)

    def _preprocess(self, text):
        """ Preprocess text before encoding as a tensor. """
        if self.lower:
            text = text.lower()
        text = text.rstrip('\n')
        if self.tokenize:
            text = self.tokenize(text)
        return text

    def encode(self, text):
        text = self._preprocess(text)
        vector = [self.stoi.get(token, UNKNOWN_INDEX) for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        return NotImplementedError
