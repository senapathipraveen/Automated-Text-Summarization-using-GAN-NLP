import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt=None, max_len=15):
        # Encode
        embedded_src = self.embed(src)
        _, (hidden, cell) = self.encoder(embedded_src)
        
        batch_size = src.size(0)    
        device = src.device
        
        if tgt is not None:
            # Shift target right for Teacher Forcing during training
            dec_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            dec_input = torch.cat([dec_input, tgt[:, :-1]], dim=1)
            
            embedded_tgt = self.embed(dec_input)
            outputs, _ = self.decoder(embedded_tgt, (hidden, cell))
            return self.fc(outputs)
        else:
            # Auto-regressive decode during generation
            dec_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            outputs = []
            for _ in range(max_len):
                embedded_tgt = self.embed(dec_input)
                out, (hidden, cell) = self.decoder(embedded_tgt, (hidden, cell))
                prediction = self.fc(out) 
                outputs.append(prediction)
                dec_input = prediction.argmax(dim=-1)
            return torch.cat(outputs, dim=1)
