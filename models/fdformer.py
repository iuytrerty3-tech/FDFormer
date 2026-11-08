"""FDFormer: Frequency-Decoupled cross-attention transFormer for style transfer.

Pipeline:
    1. Frozen VGG-19 encodes content and style images.
    2. The deepest feature (relu4_1) of each is split by FreqDecompose into a
       low-frequency band and a high-frequency band that sum exactly to the feature.
    3. Two independent LinearCrossAttention branches transfer each band separately.
    4. A content-adaptive gate decides per-location how much style to inject.
    5. A light decoder reconstructs the stylised RGB image.

At inference, `alpha_low` and `alpha_high` scale the two branch contributions. In the
paper's measurements `alpha_low` sets stylisation strength, while `alpha_high` has
little visible effect and does not control brushwork.

Ablation switches (used to fill the paper's ablation table):
    freq_decouple=False  -> single attention path, no frequency split
    use_gate=False       -> uniform style injection, no content-adaptive gate
    freq_mode="fixed"    -> frozen Gaussian low-pass instead of a learned one
    freq_mode="dwt"      -> Haar wavelet split instead of a Gaussian blur
    freq_mode="equal"    -> two full-capacity branches with no frequency split (F/2 each)
    (the no-high-frequency-loss ablation lives in the loss config: w_hf_stroke=0)
"""
import torch
import torch.nn as nn

from .vgg import VGGEncoder
from .freq import FreqDecompose
from .attention import LinearCrossAttention
from .gating import ContentAdaptiveGate
from .decoder import Decoder


class FDFormer(nn.Module):
    def __init__(self, embed_dim: int = 256, heads: int = 4, pretrained_vgg: bool = True,
                 freq_decouple: bool = True, use_gate: bool = True,
                 freq_mode: str = "learnable"):
        super().__init__()
        self.encoder = VGGEncoder(pretrained=pretrained_vgg)
        deep_dim = 512  # relu4_1 channels
        # Ablation switches (full model = both True). These are saved in the
        # checkpoint cfg so stylize.py / batch_stylize.py rebuild the right graph.
        self.freq_decouple = freq_decouple
        self.use_gate = use_gate
        # Which low-pass operator splits the bands: "learnable" (default), the same
        # Gaussian held "fixed", or a Haar "dwt" split. Recorded in the checkpoint
        # config so inference rebuilds the graph the checkpoint was trained with.
        self.freq_mode = freq_mode
        self.freq = FreqDecompose(deep_dim, mode=freq_mode) if freq_decouple else None
        self.attn_low = LinearCrossAttention(deep_dim, embed_dim, heads)
        self.attn_high = LinearCrossAttention(deep_dim, embed_dim, heads) if freq_decouple else None
        self.gate = ContentAdaptiveGate(deep_dim) if use_gate else None
        self.decoder = Decoder(deep_dim)

    def transfer_feature(self, c_deep, s_deep, alpha_low=1.0, alpha_high=1.0,
                         gate_scale=1.0):
        if self.freq_decouple:
            c_low, c_high = self.freq(c_deep)
            s_low, s_high = self.freq(s_deep)
            st_low = self.attn_low(c_low, s_low)
            st_high = self.attn_high(c_high, s_high)
            stylized = alpha_low * st_low + alpha_high * st_high
        else:
            # Single-path ablation: one attention over the full feature; no per-band
            # split, so alpha_low / alpha_high do not apply.
            stylized = self.attn_low(c_deep, s_deep)
        if self.use_gate:
            g = self.gate(c_deep)
            if gate_scale != 1.0:
                # A scalar on the predicted gate, clamped back into [0, 1]. It shifts
                # how much style every location admits without touching the pattern
                # the gate predicted, and so trades content fidelity against
                # stylisation along a different line than alpha does: alpha scales the
                # transferred feature, while this changes how much of it is mixed in.
                g = (g * gate_scale).clamp(0.0, 1.0)
            fused = (1.0 - g) * c_deep + g * stylized
        else:
            # No-gate ablation: inject style uniformly across all locations.
            g = None
            fused = stylized
        return fused, g

    def forward(self, content, style, alpha_low=1.0, alpha_high=1.0, return_feat=False,
                gate_scale=1.0):
        c_feats = self.encoder(content)
        s_feats = self.encoder(style)
        fused, g = self.transfer_feature(c_feats[-1], s_feats[-1], alpha_low, alpha_high,
                                         gate_scale)
        out = self.decoder(fused)
        if return_feat:
            return out, {"target": fused, "gate": g,
                         "content_feats": c_feats, "style_feats": s_feats}
        return out

    @torch.no_grad()
    def stylize_low_memory(self, content, style, alpha_low=1.0, alpha_high=1.0, gate_scale=1.0):
        """Same output as forward(), holding only relu4_1 of each input.

        forward() keeps all four encoder maps of both images alive for the loss, which
        dominates peak memory at high resolution. Inference needs only the deepest map.
        """
        c_deep = self.encoder.encode_deepest(content)
        s_deep = self.encoder.encode_deepest(style)
        fused, _ = self.transfer_feature(c_deep, s_deep, alpha_low, alpha_high, gate_scale)
        del c_deep, s_deep
        return self.decoder(fused)

    def trainable_parameters(self):
        """Everything except the frozen VGG encoder."""
        return [p for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("encoder.")]
