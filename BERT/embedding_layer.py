import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_size, segment_vocab_size, dropout_prob=0.1):
        super(EmbeddingLayer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEncoding (hidden_size, dropout_prob, max_seq_len)
        self.segment_embeddings = nn.Embedding(segment_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, segment_ids):
        token_embeddings = self.token_embeddings(input_ids)
        # Transpose to [seq_len, batch_size, embedding_dim]
        token_embeddings = token_embeddings.transpose(0, 1)

        position_embeddings = self.position_embeddings(token_embeddings)
        segment_embeddings = self.segment_embeddings(segment_ids)
        
        embeddings = token_embeddings.transpose(0, 1) + position_embeddings.transpose(0,1) + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
