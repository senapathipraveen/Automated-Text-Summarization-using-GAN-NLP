
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim,1)

    def forward(self,x):
        x = self.embed(x).mean(dim=1)
        return torch.sigmoid(self.fc(x))
