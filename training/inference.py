"""Shared inference: features -> per-frame head probabilities -> Harte labels -> segments."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from chord_vocab import decode
from features import FRAME_SECONDS, LOG_FLOOR

METRICS = ["root", "majmin", "majmin_inv", "sevenths", "sevenths_inv", "tetrads", "tetrads_inv", "mirex"]


@torch.no_grad()
def predict_probs(model, feats: np.ndarray, mean: float, std: float, device, batch: int = 32) -> dict:
    """Run overlapping windows (hop = half a window) over a song and average the probabilities.

    feats: raw log-CQT [T, 144]. Returns {head: np.ndarray [T, C]} (softmax; sigmoid for ext).
    """
    model.eval()
    ts = model.timestep
    hop = ts // 2
    n = len(feats)
    starts = list(range(0, max(n - ts, 0) + 1, hop))
    if starts[-1] + ts < n:
        starts.append(n - ts)
    x = np.full((len(starts), ts, feats.shape[1]), LOG_FLOOR, np.float32)
    for i, s in enumerate(starts):
        chunk = feats[s:s + ts]
        x[i, :len(chunk)] = chunk
    x = (x - mean) / std

    sums: dict[str, np.ndarray] = {}
    counts = np.zeros(n, np.float32)
    # Down-weight window edges, where the model has the least context.
    w = np.hanning(ts + 2)[1:-1].astype(np.float32) + 1e-3
    for b in range(0, len(starts), batch):
        out = model(torch.from_numpy(x[b:b + batch]).to(device))
        for name, logits in out.items():
            p = (torch.sigmoid(logits) if name == "ext" else torch.softmax(logits, -1)).cpu().numpy()
            acc = sums.setdefault(name, np.zeros((n, p.shape[-1]), np.float32))
            for j, s in enumerate(starts[b:b + batch]):
                m = min(ts, n - s)
                acc[s:s + m] += p[j, :m] * w[:m, None]
                if name == "root":
                    counts[s:s + m] += w[:m]
    return {k: v / counts[:, None] for k, v in sums.items()}


def decode_frames(probs: dict, ext_threshold: float = 0.5) -> list[str]:
    root = probs["root"].argmax(-1)
    triad = probs["triad"].argmax(-1)
    seventh = probs["seventh"].argmax(-1)
    bass = probs["bass"].argmax(-1)
    ext = probs["ext"] >= ext_threshold
    cache: dict[tuple, str] = {}
    out = []
    for i in range(len(root)):
        key = (int(root[i]), int(triad[i]), int(seventh[i]), tuple(ext[i].tolist()), int(bass[i]))
        if key not in cache:
            cache[key] = decode(*key)
        out.append(cache[key])
    return out


def frames_to_segments(labels: list[str], duration: float | None = None,
                       min_duration: float = 0.0) -> list[tuple[float, float, str]]:
    segs: list[list] = []
    for i, lab in enumerate(labels):
        if segs and segs[-1][2] == lab:
            segs[-1][1] = (i + 1) * FRAME_SECONDS
        else:
            segs.append([i * FRAME_SECONDS, (i + 1) * FRAME_SECONDS, lab])
    if min_duration > 0 and len(segs) > 1:
        # Absorb too-short segments into the previous one (or the next, at the start).
        changed = True
        while changed and len(segs) > 1:
            changed = False
            for i, s in enumerate(segs):
                if s[1] - s[0] < min_duration:
                    if i > 0:
                        segs[i - 1][1] = s[1]
                    else:
                        segs[1][0] = s[0]
                    del segs[i]
                    changed = True
                    break
            merged = [segs[0]]
            for s in segs[1:]:
                if s[2] == merged[-1][2]:
                    merged[-1][1] = s[1]
                else:
                    merged.append(s)
            segs = merged
    if duration is not None and segs:
        segs[-1][1] = duration
    return [(a, b, c) for a, b, c in segs]


def frame_metrics(ref: list[str], est: list[str]) -> dict[str, float]:
    """MIREX-style chord metrics over frames (ref X frames excluded). Counts unique pairs once."""
    import mir_eval
    pairs = Counter((r, e) for r, e in zip(ref, est) if r != "X")
    if not pairs:
        return {m: float("nan") for m in METRICS + ["exact"]}
    refs = [p[0] for p in pairs]
    ests = ["N" if p[1] == "X" else p[1] for p in pairs]
    weights = np.array([pairs[p] for p in pairs], float)
    out = {}
    for m in METRICS:
        comp = np.asarray(getattr(mir_eval.chord, m)(refs, ests), float)
        ok = comp >= 0
        out[m] = float((comp[ok] * weights[ok]).sum() / weights[ok].sum()) if ok.any() else float("nan")
    out["exact"] = float(sum(w for (r, e), w in zip(pairs, weights) if r == e) / weights.sum())
    return out
