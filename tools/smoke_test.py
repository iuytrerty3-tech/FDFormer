"""CPU smoke test: validates wiring without a GPU, dataset, or pretrained download.

Builds a tiny FDFormer (random VGG weights), runs a forward pass on random tensors,
checks output shape, exercises the alpha knobs, then runs one loss + backward step and
confirms gradients reach the trainable parameters. It does not measure quality.

    python tools/smoke_test.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FDFormer
from losses.losses import FDFormerCriterion

WEIGHTS = {"w_content": 1.0, "w_style": 3.0, "w_identity1": 50.0,
           "w_identity2": 1.0, "w_hf_stroke": 1.0}


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = FDFormer(embed_dim=64, heads=2, pretrained_vgg=False).to(device)

    content = torch.rand(2, 3, 64, 64)
    style = torch.rand(2, 3, 64, 64)

    out = model(content, style)
    assert out.shape == (2, 3, 64, 64), f"bad output shape {out.shape}"
    print("forward OK            ->", tuple(out.shape))

    o_lo = model(content, style, alpha_low=0.3, alpha_high=1.0)
    o_hi = model(content, style, alpha_low=1.0, alpha_high=1.7)
    assert not torch.allclose(o_lo, o_hi), "alpha knobs have no effect"
    print("alpha controllability OK (low vs high outputs differ)")

    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params      -> {n_train/1e6:.2f}M  (of {n_total/1e6:.2f}M total; "
          f"VGG frozen)")

    criterion = FDFormerCriterion(model.encoder, WEIGHTS, use_identity=True)
    total, comps = criterion(model, content, style)
    total.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.trainable_parameters())
    assert grad_ok, "no finite gradients on trainable params"
    print("loss + backward OK    ->", {k: round(float(v), 4) for k, v in comps.items()})
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
