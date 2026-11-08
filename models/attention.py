"""Linear cross-attention with a structural relative position encoding.

Standard softmax cross-attention costs O(Nq * Nk). We use the kernel-feature-map
linearisation of Katharopoulos et al. (ICML 2020), phi(x) = elu(x) + 1, which brings
the cost to O(N) and lets a laptop GPU stylise higher resolutions without OOM.

Each branch matches content tokens (queries) against style tokens (keys/values) of a
single frequency band. A lightweight depthwise conv injects a structural relative
position cue before projection, extending the content-aware positional encoding idea
of StyTr2 (Deng et al., CVPR 2022).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCrossAttention(nn.Module):
    def __init__(self, in_dim: int, dim: int = 256, heads: int = 4):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim

        self.pe = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim)  # structural PE
        self.to_q = nn.Conv2d(in_dim, dim, 1)
        self.to_k = nn.Conv2d(in_dim, dim, 1)
        self.to_v = nn.Conv2d(in_dim, dim, 1)
        self.proj = nn.Conv2d(dim, in_dim, 1)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, dim, H, W] -> [B, heads, N, head_dim]
        b, d, h, w = x.shape
        return x.reshape(b, self.heads, self.head_dim, h * w).permute(0, 1, 3, 2)

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        b, c, hc, wc = content_feat.shape
        cf = content_feat + self.pe(content_feat)
        sf = style_feat + self.pe(style_feat)

        q = self._to_heads(self.to_q(cf))   # B, h, Nq, d
        k = self._to_heads(self.to_k(sf))   # B, h, Nk, d
        v = self._to_heads(self.to_v(sf))   # B, h, Nk, d

        # Cast to float64 for einsum to avoid numerical overflow in AMP/float16
        # The linear attention involves large matrix products that can overflow float16/32
        q = q.double()
        k = k.double()
        v = v.double()

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv = torch.einsum("bhnd,bhne->bhde", k, v)        # B, h, d, d
        k_sum = k.sum(dim=2)                              # B, h, d
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k_sum) + 1e-6)
        out = torch.einsum("bhnd,bhde->bhne", q, kv)      # B, h, Nq, d
        out = out * z.unsqueeze(-1)

        out = out.permute(0, 1, 3, 2).contiguous().reshape(b, self.dim, hc, wc)
        # Cast back to original dtype
        return self.proj(out.to(content_feat.dtype))
