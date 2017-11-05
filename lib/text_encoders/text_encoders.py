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
        # TODO: https://github.com/pytorch/text/blob/master/torchtext/vocab.py
        return None
