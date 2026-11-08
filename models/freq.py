"""Frequency decomposition used by FDFormer.

A feature map is split into a low-frequency band and a high-frequency band:

    low  = LowPass(F)
    high = F - low

so that F = low + high holds exactly and no information is discarded, only routed.
Each band is sent to its own cross-attention branch. The paper finds that the bands
do not separate colour from texture, and that only the low-band coefficient acts as
a usable control at inference.

Three low-pass operators are available, selected by ``mode``:

    "learnable"  depthwise blur initialised to a Gaussian and then trained (default)
    "fixed"      the same Gaussian, frozen at its initial value
    "dwt"        the LL branch of a single-level Haar wavelet transform, expanded
                 back to full resolution (a critically sampled, non-Gaussian split)
    "equal"      no frequency selectivity: each band is F/2, so low + high = F still
                 holds and the two attention branches keep their full capacity

"fixed", "dwt" and "equal" exist so that the learnable split can be compared against
non-adaptive alternatives under an otherwise identical training recipe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel2d(ksize: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g2 = torch.outer(g1, g1)
    return g2 / g2.sum()


class FreqDecompose(nn.Module):
    def __init__(self, channels: int, ksize: int = 5, sigma: float = 1.0,
                 mode: str = "learnable"):
        super().__init__()
        if mode not in ("learnable", "fixed", "dwt", "equal"):
            raise ValueError(f"unknown decomposition mode: {mode}")
        self.channels = channels
        self.ksize = ksize
        self.pad = ksize // 2
        self.mode = mode
        if mode in ("dwt", "equal"):
            # No kernel. For "dwt" the low-pass is the Haar LL branch (a 2x2 box average
            # followed by the matching expansion), which has no free parameters.
            self.weight = None
        else:
            # Depthwise kernel, one per channel, initialised to a Gaussian.
            kernel = _gaussian_kernel2d(ksize, sigma)
            weight = kernel.view(1, 1, ksize, ksize).repeat(channels, 1, 1, 1).clone()
            # Kept as a Parameter in both cases so the state dict is identical;
            # "fixed" simply never receives a gradient.
            self.weight = nn.Parameter(weight, requires_grad=(mode == "learnable"))

    def _lowpass_conv(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            F.pad(x, [self.pad] * 4, mode="reflect"),
            self.weight,
            groups=self.channels,
        )

    def _lowpass_haar(self, x: torch.Tensor) -> torch.Tensor:
        """LL band of one Haar level, expanded back to the input resolution.

        Averaging each 2x2 block and repeating the average over that block is
        exactly the reconstruction from the LL coefficients alone, so the residual
        x - low collects the LH, HL and HH coefficients.
        """
        b, c, h, w = x.shape
        ph, pw = h % 2, w % 2
        if ph or pw:
            x = F.pad(x, [0, pw, 0, ph], mode="reflect")
        low = F.avg_pool2d(x, 2)
        low = F.interpolate(low, scale_factor=2, mode="nearest")
        return low[:, :, :h, :w]

    def forward(self, x: torch.Tensor):
        if self.mode == "equal":
            low = 0.5 * x
        else:
            low = self._lowpass_haar(x) if self.mode == "dwt" else self._lowpass_conv(x)
        high = x - low
        return low, high
