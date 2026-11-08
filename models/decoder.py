"""Lightweight decoder that inverts relu4_1 features back to an RGB image.

Mirror of the VGG encoder down to image scale (the standard AdaIN/StyTr2 decoder),
using nearest-neighbour upsampling + reflection padding to avoid checkerboard
artifacts. Outputs are clamped to [0, 1] by the caller.
"""
import torch.nn as nn


def _conv(in_c, out_c):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_c, out_c, 3),
        nn.ReLU(inplace=True),
    )


class Decoder(nn.Module):
    def __init__(self, in_dim: int = 512):
        super().__init__()
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")
        self.net = nn.Sequential(
            _conv(in_dim, 256), up(),          # 1/8 -> 1/4
            _conv(256, 256), _conv(256, 256), _conv(256, 256),
            _conv(256, 128), up(),             # 1/4 -> 1/2
            _conv(128, 128),
            _conv(128, 64), up(),              # 1/2 -> 1/1
            _conv(64, 64),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.net(x)
