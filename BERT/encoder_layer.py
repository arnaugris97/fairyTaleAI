import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self,  hidden_size, num_attention_heads, intermediate_size, num_hidden_layers, dropout_prob):
        super(EncoderLayer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=intermediate_size, dropout=dropout_prob, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

    def forward(self, hidden_states, attention_mask=None):

        attention_mask = attention_mask.to(dtype=torch.bool)
        
        encoder_output = self.encoder(hidden_states, src_key_padding_mask=attention_mask, is_causal=False)

        return encoder_output