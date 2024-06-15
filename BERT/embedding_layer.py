import math
import torch
import torch.nn as nn

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
        print(self.pe.shape)

    def forward(self, x):
        return self.pe
    
class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        super(EmbeddingLayer, self).__init__()
    
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment_embeddings = nn.Embedding(3, embed_size, padding_idx=0)
        self.position_embeddings = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = nn.Dropout(p=dropout)
       
    def forward(self, input_ids, segment_ids):
        x = self.token_embeddings(input_ids) + self.position_embeddings(input_ids) + self.segment_embeddings(segment_ids)
        x = self.dropout(x)
        return x
