"""BTC backbone with factorised output heads.

The backbone is BTC's bi-directional self-attention stack, initialised from the pretrained
large-vocabulary checkpoint. The 170-class softmax is replaced by one head per chord
component (see chord_vocab.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import btc_compat
from chord_vocab import HEAD_SIZES, IGNORE

BACKBONE_DEFAULTS = dict(feature_size=144, hidden_size=128, num_layers=8, num_heads=4,
                         total_key_depth=128, total_value_depth=128, filter_size=128,
                         timestep=108, input_dropout=0.2, layer_dropout=0.2,
                         attention_dropout=0.2, relu_dropout=0.2)


class FactorisedBTC(nn.Module):
    def __init__(self, head_dropout: float = 0.1, **backbone):
        super().__init__()
        cfg = {**BACKBONE_DEFAULTS, **backbone}
        self.timestep = cfg["timestep"]
        self.backbone = btc_compat.bi_directional_self_attention_layers(
            cfg["feature_size"], cfg["hidden_size"], cfg["num_layers"], cfg["num_heads"],
            cfg["total_key_depth"], cfg["total_value_depth"], cfg["filter_size"], cfg["timestep"],
            cfg["input_dropout"], cfg["layer_dropout"], cfg["attention_dropout"], cfg["relu_dropout"])
        self.dropout = nn.Dropout(head_dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(cfg["hidden_size"], size)
                                    for name, size in HEAD_SIZES.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [batch, frames <= timestep, 144] normalised features -> logits per head."""
        h, _ = self.backbone(x)
        h = self.dropout(h)
        return {name: head(h) for name, head in self.heads.items()}

    def load_pretrained_backbone(self, checkpoint: dict) -> int:
        """Copy BTC's self_attn_layers.* weights into the backbone. Returns tensors loaded."""
        prefix = "self_attn_layers."
        state = {k[len(prefix):]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)}
        self.backbone.load_state_dict(state, strict=True)
        return len(state)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return list(self.heads.parameters())


class FactorisedLoss(nn.Module):
    """Sum of per-head losses. Cross-entropy (optionally focal / class-weighted) for the
    single-label heads, masked binary cross-entropy for the extension head."""

    def __init__(self, head_weights: dict[str, float], class_weights: dict[str, torch.Tensor] | None = None,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.head_weights = head_weights
        self.focal_gamma = focal_gamma
        for name, w in (class_weights or {}).items():
            self.register_buffer(f"cw_{name}", w)

    def _ce(self, logits, target, name):
        weight = getattr(self, f"cw_{name}", None)
        logits = logits.reshape(-1, logits.shape[-1])
        target = target.reshape(-1)
        valid = target != IGNORE
        if not valid.any():
            return logits.sum() * 0.0
        logits, target = logits[valid], target[valid]
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        if self.focal_gamma > 0:
            p = torch.softmax(logits, -1).gather(1, target[:, None]).squeeze(1)
            ce = ce * (1 - p) ** self.focal_gamma
        if weight is not None:
            return ce.sum() / weight[target].sum()
        return ce.mean()

    def forward(self, logits: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        parts = {}
        for name in ("root", "triad", "seventh", "bass"):
            parts[name] = self._ce(logits[name], batch[name], name)
        mask = batch["ext_mask"]
        if mask.sum() > 0:
            bce = F.binary_cross_entropy_with_logits(logits["ext"], batch["ext"], reduction="none")
            parts["ext"] = (bce.mean(-1) * mask).sum() / mask.sum()
        else:
            parts["ext"] = logits["ext"].sum() * 0.0
        total = sum(self.head_weights.get(k, 1.0) * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
