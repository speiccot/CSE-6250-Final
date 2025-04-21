import functools
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeprLayer(nn.Module):
    def __init__(self, feature_size: int = 100, window: int = 1, hidden_size: int = 3):
        super(DeeprLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=feature_size,
            out_channels=hidden_size,
            kernel_size=2 * window + 1,
            padding=window
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = F.relu(self.conv(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        return x


def _flatten_and_fill_gap(gap_embedding, batch, device):
    embed_dim = gap_embedding.shape[-1]
    batch = [
        [
            [torch.tensor(vec, device=device, dtype=torch.float) for vec in visit]
            for visit in patient
        ]
        for patient in batch
    ]
    batch = [
        torch.stack(functools.reduce(lambda a, b: a + [gap_embedding] + b, patient), 0)
        for patient in batch
    ]
    batch_max_len = max(len(seq) for seq in batch)
    mask = torch.tensor([
        [1] * len(seq) + [0] * (batch_max_len - len(seq)) for seq in batch
    ], dtype=torch.long, device=device)

    out = torch.zeros((len(batch), batch_max_len, embed_dim), device=device)
    for i, seq in enumerate(batch):
        out[i, :len(seq)] = seq
    return out, mask


class Deepr(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 1,
        window: int = 1,
    ):
        super(Deepr, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.deepr_layer = DeeprLayer(feature_size=embedding_dim, window=window, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = self.deepr_layer(x, mask)  # [batch, hidden_dim]
        logits = self.fc(x)  # [batch, output_dim]
        return logits


# Example usage (remove/comment out for module usage)
if __name__ == "__main__":
    batch_size = 4
    seq_len = 20
    vocab_size = 1000

    model = Deepr(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones_like(x, dtype=torch.long)

    logits = model(x, mask)
    print("Logits shape:", logits.shape)  # [batch_size, 1]