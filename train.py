import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import TextSummaryDataset

EPOCHS = 150
BATCH_SIZE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    docs = [item[0] for item in batch]
    sums = [item[1] for item in batch]
    docs_pad = pad_sequence(docs, batch_first=True, padding_value=0)
    sums_pad = pad_sequence(sums, batch_first=True, padding_value=0)
    max_len = max(docs_pad.size(1), sums_pad.size(1))
    docs_pad = F.pad(docs_pad, (0, max_len - docs_pad.size(1)))
    sums_pad = F.pad(sums_pad, (0, max_len - sums_pad.size(1)))
    return docs_pad, sums_pad

dataset = TextSummaryDataset('data/train.csv')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
vocab_size = len(dataset.vocab)

G = Generator(vocab_size).to(DEVICE)
D = Discriminator(vocab_size).to(DEVICE)

g_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
d_opt = torch.optim.Adam(D.parameters(), lr=0.0002)

bce = nn.BCELoss()
ce = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for doc, real_sum in loader:
        doc, real_sum = doc.to(DEVICE), real_sum.to(DEVICE)

        fake_logits = G(doc, real_sum)
        fake_sum = fake_logits.argmax(dim=-1)

        real_lbl = torch.ones(doc.size(0),1).to(DEVICE)
        fake_lbl = torch.zeros(doc.size(0),1).to(DEVICE)

        d_loss = bce(D(real_sum), real_lbl) + bce(D(fake_sum.detach()), fake_lbl)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        g_loss = ce(fake_logits.view(-1, vocab_size), real_sum.view(-1)) +                  0.1 * bce(D(fake_sum), real_lbl)

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    print(f"Epoch {epoch+1} | D Loss {d_loss.item():.4f} | G Loss {g_loss.item():.4f}")

torch.save(G.state_dict(), 'generator.pth')
