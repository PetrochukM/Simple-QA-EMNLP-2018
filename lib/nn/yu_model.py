import torch
import torch.nn as nn

from lib.text_encoders import PADDING_INDEX


class YuModel(nn.Module):

    def __init__(self,
                 relation_vocab_size,
                 relation_word_vocab_size,
                 text_vocab_size,
                 embedding_size=300,
                 hidden_size=200):
        super().__init__()

        self.relation_embedding = nn.Embedding(
            relation_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.relation_word_embedding = nn.Embedding(
            relation_word_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.relation_word_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0,
            bidirectional=True)
        self.relation_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0,
            bidirectional=True)

        self.text_embedding = nn.Embedding(
            text_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.text_embedding.weight.requires_grad = False
        self.text_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0,
            bidirectional=True)

        self.distance = nn.CosineSimilarity(dim=1)

    def forward(self, text, relation, relation_word):
        """
        Args:
            text: (torch.LongTensor [text_len, batch_size])
            relation: (torch.LongTensor [relation_len, batch_size])
            relation_word: (torch.LongTensor [relation_word_len, batch_size])
        """
        batch_size = text.size()[1]

        relation_word = self.relation_word_embedding(relation_word)
        _, (relation_word, _) = self.relation_word_rnn(relation_word)
        relation_word = relation_word[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        relation = self.relation_embedding(relation)
        _, (relation, _) = self.relation_rnn(relation)
        relation = relation[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        # (2, batch_size, hidden_size * 2)
        relation = torch.stack([relation, relation_word])
        # (batch_size, hidden_size * 2)
        relation = torch.max(relation, dim=0)[0]

        text = self.text_embedding(text)
        _, (text, _) = self.text_rnn(text)
        text = text[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        # (batch_size)
        return self.distance(text, relation)