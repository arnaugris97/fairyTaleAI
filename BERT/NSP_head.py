import torch.nn as nn

class NSPHead(nn.Module):
    def __init__(self, hidden_size):
        super(NSPHead, self).__init__()
        self.cls = nn.Linear(hidden_size, 1)

    def forward(self, pooled_output):
        seq_relationship_scores = self.cls(pooled_output)
        return seq_relationship_scores