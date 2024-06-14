import math
import torch
import torch.nn as nn
from torch import Tensor

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]

class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe  
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, batch_size, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, batch_size, d_model)
#         pe[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(1)
#         pe[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(1)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0), :, :]
#         return x

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_size, segment_vocab_size, dropout_prob=0.1 ):
        super(EmbeddingLayer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size,  padding_idx=0)
        self.position_embeddings = PositionalEmbedding (hidden_size,  max_seq_len)
        self.segment_embeddings = nn.Embedding(segment_vocab_size, hidden_size,  padding_idx=0)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, segment_ids):
        token_embeddings = self.token_embeddings(input_ids)
        # Transpose to [seq_len, batch_size, embedding_dim]
        token_embeddings = token_embeddings.transpose(0, 1)

        position_embeddings = self.position_embeddings(token_embeddings)
        segment_embeddings = self.segment_embeddings(segment_ids)
        
        embeddings = token_embeddings.transpose(0, 1) + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
