import logging

import torch.nn as nn

from lib.nn.lock_dropout import LockedDropout
from lib.text_encoders import PADDING_INDEX

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

    def __init__(self, vocab_size, embeddings, embedding_size, embedding_dropout, rnn_dropout,
                 rnn_cell, freeze_embeddings):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            self.vocab_size, int(embedding_size), padding_idx=PADDING_INDEX)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.rnn_dropout = LockedDropout(p=rnn_dropout)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        if embeddings is not None:
            logger.info('Loading embeddings...')
            self.embedding.weight.data.copy_(embeddings)

    def forward(self, *args, **kwargs):

        raise NotImplementedError()
