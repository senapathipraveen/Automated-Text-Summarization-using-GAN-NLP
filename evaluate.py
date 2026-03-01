import torch
from models.generator import Generator
from utils.dataset import TextSummaryDataset

dataset = TextSummaryDataset('data/train.csv')
G = Generator(len(dataset.vocab))
G.load_state_dict(torch.load('generator.pth'))
G.eval()

doc, real_sum = dataset[0]
out = G(doc.unsqueeze(0))
pred_tokens = out.argmax(dim=-1).squeeze(0).tolist()

inv_vocab = {v: k for k, v in dataset.vocab.items()}
pred_words = [inv_vocab.get(t, '<UNK>') for t in pred_tokens if t != 0]

print("Predicted Summary tokens:", pred_tokens)
print("Predicted Summary words:", pred_words)
