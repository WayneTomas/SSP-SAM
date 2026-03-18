import torch
import torch.nn as nn
import torch.nn.functional as F


class PhraseAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, mask):
        cxt_scores = self.fc(context).squeeze(2)

        attn = F.softmax(cxt_scores, dim=-1)

        attn = attn * ((~mask).float())  # (batch, seq_len)

        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)

        attn3 = attn.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embedded)
        return attn, weighted_emb
