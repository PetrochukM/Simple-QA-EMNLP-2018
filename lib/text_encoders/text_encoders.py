import torch

from lib.text_encoders.subword_text_tokenizer import SubwordTextTokenizer

# RESERVED TOKENS
# NOTE: vocab size is len(reversed) + len(vocab)
PADDING_INDEX = 0
UNKNOWN_INDEX = 1
EOS_INDEX = 2
SOS_INDEX = 3
PADDING_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
DEFAULT_ITOS = [PADDING_TOKEN, UNKNOWN_TOKEN, EOS_TOKEN, SOS_TOKEN]
DEFAULT_STOI = {token: index for index, token in enumerate(DEFAULT_ITOS)}


class TextEncoder(object):

    def __init__(self):
        raise NotImplementedError

    def encode(self, string):
        """ Given a string encode it into a tensor """
        raise NotImplementedError

    def decode(self, tensor):
        """ Given a tensor decode it into a string """
        raise NotImplementedError

    @property
    def vocab_size(self):
        """ Return the size of the vocab used to encode the text """
        raise NotImplementedError

    @property
    def embeddings(self):
        """ Return an embedding tensor such that for each index 0 to vocab_size there exists an
        embedding """
        return None


class StaticTokenizerEncoder(TextEncoder):

    def __init__(self, sample, append_eos=True, lower=True, tokenize=(lambda s: s.split())):
        """ Given a sample, build the dictionary for the word encoder """
        self.lower = lower
        self.tokenize = tokenize
        self.append_eos = append_eos
        self.vocab = set()

        for text in sample:
            self.vocab.update(self._preprocess(text))

        self.stoi = DEFAULT_STOI.copy()
        self.itos = DEFAULT_ITOS[:]
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
        vector = [self.stoi[token] for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        return NotImplementedError


class WordEncoder(StaticTokenizerEncoder):

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)


class CharacterEncoder(StaticTokenizerEncoder):

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('CharacterEncoder defines a tokenize callable per character')
        super().__init__(*args, **kwargs, tokenize=(lambda s: s.split('')))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ''.join(tokens)


class MosesEncoder(WordEncoder):

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('MosesEncoder defines a tokenize callable Moses')

        import nltk

        # Required for moses
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')

        from nltk.tokenize.moses import MosesTokenizer

        super().__init__(*args, **kwargs, tokenize=MosesTokenizer().tokenize)


class IdentityEncoder(StaticTokenizerEncoder):

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('IdentityEncoder defines a identity tokenize s -> [s]')
        if 'append_eos' not in kwargs:
            kwargs['append_eos'] = False  # Default to not appending EOS
        super().__init__(*args, **kwargs, tokenize=(lambda s: [s]))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ''.join(tokens)


class SubwordEncoder(TextEncoder):

    def __init__(
            self,
            sample,
            append_eos=True,
            lower=True,
            target_size=None,
            min_val=1,
            max_val=1e3,):
        """ Given a sample, build the dictionary for the word encoder.
        
        Args:
            sample (list of str)
            append_eos (bool)
            lower (bool)
            target_size (int): desired vocab_size to approximate
            min_val (int): lower bound for the minimum token count
            max_val (int): upper bound for the minimum token count
        """
        self.lower = lower
        self.append_eos = append_eos

        if self.lower:
            sample = [text.lower().rstrip('\n') for text in sample]

        if target_size is None:
            self.tokenizer = SubwordTextTokenizer()
            self.tokenizer.build_from_corpus(sample, min_val=min_val)
        else:
            self.tokenizer = SubwordTextTokenizer.build_to_target_size_from_corpus(
                sample, target_size=target_size, min_val=min_val, max_val=max_val)

        self.stoi = DEFAULT_STOI.copy()
        self.itos = DEFAULT_ITOS[:]
        for token in self.tokenizer.vocab:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, text):
        if self.lower:
            text = text.lower()
        text = text.rstrip('\n')
        text = self.tokenizer.encode(text)
        vector = [self.stoi[token] for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.tokenizer.decode(tokens)
