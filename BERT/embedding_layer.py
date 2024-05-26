import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_size, segment_vocab_size):
        super(EmbeddingLayer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

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
