import torch
import torch.nn as nn

class LangIDModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_units, num_classes, dropout, num_layers):
        super(LangIDModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(gru_units * 2, num_classes)

    def forward(self,x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
            
        return x
