import logging

import torch.nn as nn

from seq2seq.fields import SeqField
from seq2seq.models.lock_dropout import LockedDropout

logger = logging.getLogger(__name__)


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab (Vocabulary): an object of Vocabulary class
        embedding_size (int): the size of the embedding
        embedding_dropout (float, optional): dropout probability an embedding (default: 0)
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')
    """

    def __init__(self, vocab, embedding_size, embedding_dropout, rnn_dropout, rnn_cell,
                 freeze_embeddings):
        super(BaseRNN, self).__init__()
        embedding_size = int(embedding_size)
        self.vocab = vocab
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.padding_idx = self.vocab.stoi[SeqField.PAD_TOKEN]
        self.rnn_dropout = LockedDropout(p=rnn_dropout)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.embedding = nn.Embedding(len(self.vocab), embedding_size, padding_idx=self.padding_idx)
        if hasattr(self.vocab, 'vectors') and self.vocab.vectors is not None:
            logger.info('Loading embeddings...')
            assert self.vocab.vectors.size()[0] == len(self.vocab), """Vocab size must be the same
                  as the number of self.vocab.vectors"""
            assert self.vocab.vectors.size()[
                1] == embedding_size, """Embedding size has to be the same size as the
                  pretrained embeddings."""
            self.embedding.weight.data.copy_(self.vocab.vectors)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

    def forward(self, *args, **kwargs):

        raise NotImplementedError()
