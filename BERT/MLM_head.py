import torch
import torch.nn as nn

class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(MLMHead, self).__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        prediction_scores = self.decoder(hidden_states)
        return prediction_scores