"""Training objective for FDFormer.

Combines the standard arbitrary-style-transfer terms with one addition tied to our
frequency decoupling:

    L = w_c * L_content              (AdaIN-style feature matching)
      + w_s * L_style               (mean/std matching over relu1_1..relu4_1)
      + w_id1 * L_identity_image    (StyTr2-style identity, image space)
      + w_id2 * L_identity_feat     (StyTr2-style identity, feature space)
      + w_hf * L_hf_stroke          (high-frequency Gram matching)

L_hf_stroke matches the Gram statistics of the *high-frequency* band of the output to
that of the style image, as a direct supervision signal for the
high-frequency attention branch is responsible for.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mean_std(feat, eps: float = 1e-5):
    b, c = feat.shape[:2]
    flat = feat.reshape(b, c, -1)
    mean = flat.mean(dim=2).view(b, c, 1, 1)
    std = (flat.var(dim=2) + eps).sqrt().view(b, c, 1, 1)
    return mean, std


def style_loss(out_feats, style_feats):
    loss = out_feats[0].new_zeros(())
    for o, s in zip(out_feats, style_feats):
        om, osd = calc_mean_std(o)
        sm, ssd = calc_mean_std(s)
        loss = loss + F.mse_loss(om, sm) + F.mse_loss(osd, ssd)
    return loss


def style_loss_lerp(out_feats, content_feats, style_feats, alpha):
    """Style term whose target moves with the coefficient.

    The plain style term asks for the painting's channel statistics whatever the
    blend coefficient is, which rewards a decoder that ignores the coefficient
    entirely. Here the target is placed between the photograph's statistics and the
    painting's, at the fraction the coefficient asks for: at alpha = 0 the output is
    asked to keep the content's own statistics, at alpha = 1 to reach the style's,
    and beyond 1 to overshoot in the same direction. Rendering each setting
    faithfully is then the way to lower the loss, which is what the fixed-target
    version failed to reward.
    """
    loss = out_feats[0].new_zeros(())
    for o, c, s in zip(out_feats, content_feats, style_feats):
        om, osd = calc_mean_std(o)
        cm, csd = calc_mean_std(c)
        sm, ssd = calc_mean_std(s)
        tm = cm + alpha * (sm - cm)
        tsd = csd + alpha * (ssd - csd)
        loss = loss + F.mse_loss(om, tm) + F.mse_loss(osd, tsd)
    return loss


def gram(feat):
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    # Use float32 for bmm to avoid float16 overflow in AMP
    orig_dtype = f.dtype
    f = f.float()
    # Clamp features before bmm to prevent inf/nan from large activations
    f = f.clamp(-1e4, 1e4)
    result = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
    return result.to(orig_dtype)


def _gaussian_weight(channels, ksize, sigma, device, dtype):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2.view(1, 1, ksize, ksize).repeat(channels, 1, 1, 1)


def high_freq(x, ksize: int = 5, sigma: float = 1.0):
    pad = ksize // 2
    # Cast to float32 for numerical stability in AMP
    orig_dtype = x.dtype
    x = x.float()
    w = _gaussian_weight(x.shape[1], ksize, sigma, x.device, x.dtype)
    low = F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), w, groups=x.shape[1])
    result = x - low
    return result.to(orig_dtype)


def hf_stroke_loss(out_feats, style_feats, layers=(2, 3)):
    loss = out_feats[0].new_zeros(())
    for i in layers:
        og = gram(high_freq(out_feats[i]))
        sg = gram(high_freq(style_feats[i]))
        term = F.mse_loss(og, sg)
        # Guard against rare inf/nan on pathological batches
        if torch.isfinite(term):
            loss = loss + term
    return loss


def hf_stroke_loss_lerp(out_feats, content_feats, style_feats, alpha, layers=(2, 3)):
    """Stroke term whose target moves with the coefficient, as above.

    The Gram matrix of the high-pass band is interpolated between the content's and
    the style's, so that raising the stroke coefficient genuinely asks for more of
    the painting's mark-making rather than for the same fixed amount.
    """
    loss = out_feats[0].new_zeros(())
    for i in layers:
        og = gram(high_freq(out_feats[i]))
        cg = gram(high_freq(content_feats[i]))
        sg = gram(high_freq(style_feats[i]))
        term = F.mse_loss(og, cg + alpha * (sg - cg))
        if torch.isfinite(term):
            loss = loss + term
    return loss


class FDFormerCriterion(nn.Module):
    def __init__(self, encoder, weights: dict, use_identity: bool = True):
        super().__init__()
        self.encoder = encoder          # shared frozen VGG (not optimised)
        self.w = weights
        self.use_identity = use_identity

    def forward(self, model, content, style, alpha_low: float = 1.0,
                alpha_high: float = 1.0):
        """One training step's losses, at a given setting of the two coefficients.

        The coefficients default to 1, which reproduces the original objective. When
        training samples them instead, the content term is what carries the effect:
        it matches the output's deep feature to the fused target, and that target is
        built from the sampled coefficients. The decoder is therefore asked to render
        the whole family of targets rather than the single one it would otherwise
        ever see, which is what makes the coefficients usable at inference.
        """
        out, aux = model(content, style, alpha_low=alpha_low, alpha_high=alpha_high,
                         return_feat=True)
        out_feats = self.encoder(out)
        c_feats = aux["content_feats"]
        s_feats = aux["style_feats"]

        l_content = F.mse_loss(out_feats[-1], aux["target"])

        # With fixed targets the style and stroke terms ask for the same thing at
        # every coefficient, so a decoder that ignores the coefficients satisfies
        # them as well as one that honours them. Moving the targets with the
        # coefficients removes that escape. Off by default, which reproduces the
        # original objective exactly.
        if self.w.get("alpha_dependent_targets", False):
            l_style = style_loss_lerp(out_feats, c_feats, s_feats, alpha_low)
            l_hf = (hf_stroke_loss_lerp(out_feats, c_feats, s_feats, alpha_high)
                    if self.w.get("w_hf_stroke", 0) != 0 else out.new_zeros(()))
        else:
            l_style = style_loss(out_feats, s_feats)
            # no-high-frequency-loss ablation = set w_hf_stroke to 0 (skips the term)
            l_hf = (hf_stroke_loss(out_feats, s_feats)
                    if self.w.get("w_hf_stroke", 0) != 0 else out.new_zeros(()))

        l_id1 = out.new_zeros(())
        l_id2 = out.new_zeros(())
        if self.use_identity:
            out_cc = model(content, content)
            out_ss = model(style, style)
            l_id1 = F.mse_loss(out_cc, content) + F.mse_loss(out_ss, style)
            fcc, fss = self.encoder(out_cc), self.encoder(out_ss)
            for i in range(len(c_feats)):
                l_id2 = l_id2 + F.mse_loss(fcc[i], c_feats[i]) + F.mse_loss(fss[i], s_feats[i])

        total = (self.w["w_content"] * l_content
                 + self.w["w_style"] * l_style
                 + self.w["w_identity1"] * l_id1
                 + self.w["w_identity2"] * l_id2
                 + self.w["w_hf_stroke"] * l_hf)

        comps = {"content": l_content.detach(), "style": l_style.detach(),
                 "id1": l_id1.detach(), "id2": l_id2.detach(),
                 "hf": l_hf.detach(), "total": total.detach()}
        return total, comps
