"""Frozen VGG-19 encoder.

Exposes relu1_1, relu2_1, relu3_1, relu4_1 feature maps, following the standard
arbitrary-style-transfer setup (Huang & Belongie, ICCV 2017; Deng et al., CVPR 2022).
The encoder is frozen; only the FDFormer modules and decoder are trained, which
keeps the trainable footprint small enough for a laptop GPU.

Memory note: we skip the VGG classifier entirely (the FC layer alone needs 392 MB
of contiguous CPU memory).  The legacy pickle path (weights_only=False) streams
tensors sequentially, making it less prone to peak-memory fragmentation than the
weights_only unpickler.
"""
import gc
import os
import torch
import torch.nn as nn
from torchvision.models.vgg import make_layers, cfgs
import torch.hub


# ImageNet statistics used to normalise inputs before the VGG.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Slice boundaries in torchvision's vgg19().features for relu1_1..relu4_1.
_SLICES = [(0, 2), (2, 7), (7, 12), (12, 21)]


def _get_cached_vgg_path():
    cached = torch.hub.get_dir() + "/checkpoints/vgg19-dcbb9e9d.pth"
    if not os.path.exists(cached):
        torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            model_dir=torch.hub.get_dir() + "/checkpoints",
        )
    return cached


def _load_features_state_dict():
    """Load only the conv-layer weights from the VGG-19 checkpoint."""
    cached = _get_cached_vgg_path()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # weights_only=False → legacy pickle path that streams data sequentially.
    # We drop classifier tensors immediately after loading to keep the
    # resident set low.  The 392 MB FC layer is loaded transiently but freed
    # before the next tensor arrives.
    full_sd = torch.load(cached, map_location="cpu", weights_only=False)
    feat_sd = {}
    for k, v in full_sd.items():
        if k.startswith("features."):
            feat_sd[k[len("features."):]] = v
    del full_sd
    gc.collect()
    return feat_sd


class VGGEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            # Build VGG-19 conv layers WITHOUT the heavy FC classifier.
            features = make_layers(cfgs["E"], batch_norm=False)
            feat_sd = _load_features_state_dict()
            features.load_state_dict(feat_sd)
            del feat_sd
            gc.collect()
        else:
            from torchvision.models import vgg19
            features = vgg19(weights=None).features

        self.blocks = nn.ModuleList(
            [nn.Sequential(*[features[i] for i in range(s, e)]) for s, e in _SLICES]
        )
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        """Return [relu1_1, relu2_1, relu3_1, relu4_1] feature maps."""
        x = self.normalize(x)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats

    @torch.no_grad()
    def encode_deepest(self, x: torch.Tensor) -> torch.Tensor:
        """relu4_1 alone, freeing each shallower map as soon as the next is computed."""
        x = self.normalize(x)
        for block in self.blocks:
            x = block(x)
        return x

    def train(self, mode: bool = True):
        # Keep VGG in eval mode permanently (frozen BN-free, but no dropout surprises).
        return super().train(False)
