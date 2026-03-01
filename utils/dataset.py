
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter

class TextSummaryDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, sep='|')
        self.vocab = self.build_vocab()

    def build_vocab(self):  
        c = Counter()
        for t in self.data['document']:
            c.update(t.split())
        for t in self.data['summary']:
            c.update(t.split())
        vocab = {w:i+2 for i,w in enumerate(c)}
        vocab['<PAD>']=0
        vocab['<UNK>']=1
        return vocab

    def encode(self, text):
        return [self.vocab.get(w,1) for w in text.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        d = torch.tensor(self.encode(self.data.iloc[idx]['document']))
        s = torch.tensor(self.encode(self.data.iloc[idx]['summary']))
        return d,s
