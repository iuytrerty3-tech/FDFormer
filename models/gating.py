"""Content-adaptive style-intensity gate.

Predicts a per-token gate g in [0, 1] from the content structure features and uses it
to modulate how much style is injected at each spatial location:

    fused = (1 - g) * content + g * stylized

Semantically salient / high-structure regions receive less style (avoids the
over-stylisation and content leakage seen with uniform-intensity methods); flat
regions receive more. The gate is tiny, adding negligible cost.
"""
import torch
import torch.nn as nn


class ContentAdaptiveGate(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, content_feat: torch.Tensor) -> torch.Tensor:
        return self.net(content_feat)  # B, 1, H, W
