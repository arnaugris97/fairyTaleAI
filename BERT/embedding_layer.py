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
        pe = torch.zeros(max_len, d_model)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        print(f'pe:', self.pe.shape)
        print(f'x:', x.shape)
        # Ensure x is reshaped to [seq_len, batch_size, embedding_dim]
        if len(x.shape) == 2:
            x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        print(f'x after addition: {x.shape}')
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
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Print inputs for debugging
        print("Embedding Layer inputs:")
        print(f"  input_ids: {input_ids}")
        print(f"  segment_ids: {segment_ids}")
        print(f"  position_ids: {position_ids}")

        # Check for NaNs in inputs
        if torch.isnan(input_ids).any() or torch.isnan(segment_ids).any() or torch.isnan(position_ids).any():
            print("NaN detected in Embedding Layer inputs")
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Check for NaNs in outputs
        if torch.isnan(embeddings).any():
            print("NaN detected in Embedding Layer outputs")

        # Print outputs for debugging
        print("Embedding Layer outputs:")
        print(f"  embeddings: {embeddings}")

        return embeddings
