"""Fine-tune BTC with factorised chord heads.

    python training/train.py                                   # uses training/config.yaml
    python training/train.py --set run_name=exp2 --set train.epochs=30 --set loss.focal_gamma=2
    python training/train.py --config my.yaml

Outputs (runs/<run_name>/): best.pt, last.pt, config.yaml, metrics.csv, splits used,
test_metrics.json and predicted .lab files for the val/test songs (predictions/<split>/),
which evaluation/evaluate.py can score against labels/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import btc_compat  # noqa: E402
from chord_vocab import EXTENSIONS, HEAD_SIZES, IGNORE, SEVENTHS, TRIADS  # noqa: E402
from dataset import EvalWindows, TrainWindows, discover_songs, load_or_create_splits  # noqa: E402
from features import FRAME_SECONDS  # noqa: E402
from inference import decode_frames, frame_metrics, frames_to_segments, predict_probs  # noqa: E402
from model import FactorisedBTC, FactorisedLoss  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for o in overrides:
        key, _, value = o.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_weights(songs, mode: str) -> dict[str, torch.Tensor]:
    """Inverse-sqrt frequency weights for triad / seventh / bass (root is balanced by pitch shift)."""
    if mode in (None, "none"):
        return {}
    out = {}
    for name in ("triad", "seventh", "bass"):
        counts = np.zeros(HEAD_SIZES[name])
        for s in songs:
            a = getattr(s.targets, name)
            a = a[a != IGNORE]
            counts += np.bincount(a, minlength=HEAD_SIZES[name])
        w = np.where(counts > 0, 1.0 / np.sqrt(counts + 1.0), 0.0)
        if w.sum() > 0:
            w = w / w[counts > 0].mean()
        out[name] = torch.tensor(w, dtype=torch.float32)
    return out


def label_stats(songs) -> str:
    """Short summary of what's in the training labels."""
    tri, sev, ext, inv, n = np.zeros(len(TRIADS)), np.zeros(len(SEVENTHS)), np.zeros(len(EXTENSIONS)), 0, 0
    for s in songs:
        t = s.targets
        tri += np.bincount(t.triad[t.triad != IGNORE], minlength=len(TRIADS))
        sev += np.bincount(t.seventh[t.seventh != IGNORE], minlength=len(SEVENTHS))
        ext += (t.ext * t.ext_mask[:, None]).sum(0)
        inv += int(((t.bass != IGNORE) & (t.bass != 0)).sum())
        n += int((t.triad != IGNORE).sum())
    n = max(n, 1)
    fmt = lambda names, c: " ".join(f"{k}:{100 * v / n:.0f}%" for k, v in zip(names, c) if v)  # noqa: E731
    return (f"  triads   {fmt(TRIADS, tri)}\n  sevenths {fmt(SEVENTHS, sev)}\n"
            f"  ext      {fmt(EXTENSIONS, ext) or '-'}\n  slash    {100 * inv / n:.0f}% of chord frames")


def build_model(cfg: dict, device) -> tuple[FactorisedBTC, float, float, str]:
    m = cfg["model"]
    model = FactorisedBTC(head_dropout=m["head_dropout"], input_dropout=m["input_dropout"],
                          layer_dropout=m["layer_dropout"], attention_dropout=m["attention_dropout"],
                          relu_dropout=m["relu_dropout"])
    init = str(m.get("init", "btc_large"))
    if init in ("btc_large", "btc_majmin"):
        ckpt = btc_compat.load_checkpoint(btc_compat.CKPT_LARGE if init == "btc_large" else btc_compat.CKPT_MAJMIN)
        n = model.load_pretrained_backbone(ckpt)
        mean, std, desc = float(ckpt["mean"]), float(ckpt["std"]), f"BTC {init} backbone ({n} tensors)"
    elif init == "none":
        ckpt = btc_compat.load_checkpoint(btc_compat.CKPT_LARGE)      # still use BTC's normalisation
        mean, std, desc = float(ckpt["mean"]), float(ckpt["std"]), "random init"
    else:
        ckpt = torch.load(resolve(init), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        mean, std, desc = ckpt["mean"], ckpt["std"], f"checkpoint {init}"
    return model.to(device), mean, std, desc


def make_scheduler(opt, steps_per_epoch: int, epochs: int, warmup_epochs: float):
    total = max(steps_per_epoch * epochs, 1)
    warm = max(int(steps_per_epoch * warmup_epochs), 1)

    def f(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(total - warm, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, songs, mean, std, device, cfg, loss_fn=None) -> dict:
    """Frame-level chord metrics on whole songs (+ windowed loss if loss_fn is given)."""
    out = {}
    if loss_fn is not None:
        model.eval()
        loader = DataLoader(EvalWindows(songs, mean, std, model.timestep), batch_size=cfg["train"]["batch_size"])
        total, n = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            loss, _ = loss_fn(model(batch["x"]), batch)
            total += float(loss) * len(batch["x"])
            n += len(batch["x"])
        out["loss"] = total / max(n, 1)
    refs, ests = [], []
    for s in songs:
        probs = predict_probs(model, s.feats, mean, std, device)
        ests += decode_frames(probs, cfg["predict"]["ext_threshold"])
        refs += list(s.targets.labels)
    out.update(frame_metrics(refs, ests))
    return out


def write_predictions(model, songs, mean, std, device, cfg, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in songs:
        labels = decode_frames(predict_probs(model, s.feats, mean, std, device), cfg["predict"]["ext_threshold"])
        segs = frames_to_segments(labels, len(s.feats) * FRAME_SECONDS, cfg["predict"]["min_duration"])
        with open(out_dir / f"{s.song_id}.lab", "w") as f:
            for a, b, lab in segs:
                f.write(f"{a:.3f}\t{b:.3f}\t{lab}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    tc = cfg["train"]
    seed_everything(tc["seed"])
    device = pick_device(tc["device"])

    run_dir = resolve("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Data
    sources = [{"audio": resolve(s["audio"]), "labels": resolve(s["labels"])} for s in cfg["data"]["sources"]]
    songs = discover_songs(sources)
    if not songs:
        print("No songs found: need audio and a matching .lab in each data source:\n  " +
              "\n  ".join(f"{s['audio']} + {s['labels']}" for s in sources), file=sys.stderr)
        return 1
    d = cfg["data"]
    splits = load_or_create_splits(resolve(d["splits_file"]), sorted(songs), d["val_fraction"],
                                   d["test_fraction"], d["split_seed"], d.get("test_only") or [])
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    cache = resolve(d["cache_dir"])
    print(f"device {device}; songs: {len(splits['train'])} train, {len(splits['val'])} val, "
          f"{len(splits['test'])} test, {len(splits['holdout'])} holdout")
    t0 = time.time()
    for i, sid in enumerate(sorted(songs), 1):
        songs[sid].load(cache)
        print(f"\rfeatures {i}/{len(songs)}", end="", flush=True)
    print(f" ({time.time() - t0:.0f} s)")
    train_songs = [songs[s] for s in splits["train"]]
    val_songs = [songs[s] for s in splits["val"]] or train_songs[:1]
    test_songs = [songs[s] for s in splits["test"]]
    holdout_songs = [songs[s] for s in splits["holdout"]]
    if not splits["val"]:
        print("warning: no validation songs; validating on a training song (add more songs)")
    if not train_songs:
        print("No training songs after splitting.", file=sys.stderr)
        return 1
    print("training labels:\n" + label_stats(train_songs))

    # Model
    model, mean, std, desc = build_model(cfg, device)
    print(f"model: {desc}; {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    loss_fn = FactorisedLoss(cfg["loss"]["head_weights"],
                             class_weights(train_songs, cfg["loss"]["class_weighting"]),
                             cfg["loss"]["focal_gamma"]).to(device)

    train_ds = TrainWindows(train_songs, mean, std, model.timestep, tc["windows_per_epoch"],
                            cfg["augment"]["max_shift"], cfg["augment"]["gain_db"])
    loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=False, drop_last=False,
                        num_workers=tc["num_workers"], persistent_workers=tc["num_workers"] > 0)
    opt = torch.optim.AdamW([
        {"params": list(model.backbone_parameters()), "lr": tc["lr"] * tc["backbone_lr_scale"]},
        {"params": model.head_parameters(), "lr": tc["lr"]},
    ], weight_decay=tc["weight_decay"])
    sched = make_scheduler(opt, len(loader), tc["epochs"], tc["warmup_epochs"])

    metric = tc["select_metric"]
    baseline = evaluate(model, val_songs, mean, std, device, cfg)
    print(f"epoch 0 (untrained heads): val {metric} {baseline[metric]:.3f}")
    best, best_epoch, stale = -1.0, 0, 0
    fields = ["epoch", "lr", "train_loss", "train_root", "train_triad", "train_seventh", "train_ext",
              "train_bass", "val_loss", "root", "majmin", "majmin_inv", "sevenths", "sevenths_inv",
              "tetrads", "tetrads_inv", "mirex", "exact", "seconds"]
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(fields)

    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        frozen = epoch <= tc["freeze_backbone_epochs"]
        for p in model.backbone_parameters():
            p.requires_grad_(not frozen)
        model.train()
        sums, n = {}, 0
        for batch in loader:
            batch = to_device(batch, device)
            loss, parts = loss_fn(model(batch["x"]), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            for k, v in {"loss": loss.item(), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        train = {k: v / max(n, 1) for k, v in sums.items()}
        val = evaluate(model, val_songs, mean, std, device, cfg, loss_fn)
        secs = time.time() - t0
        print(f"epoch {epoch:3d}{' (heads only)' if frozen else ''}: train {train['loss']:.3f} | "
              f"val loss {val['loss']:.3f} root {val['root']:.3f} majmin {val['majmin']:.3f} "
              f"sevenths {val['sevenths']:.3f} tetrads_inv {val['tetrads_inv']:.3f} "
              f"exact {val['exact']:.3f} | {secs:.0f}s", flush=True)
        row = {"epoch": epoch, "lr": opt.param_groups[1]["lr"], "seconds": round(secs, 1),
               **{f"train_{k}" if k != "loss" else "train_loss": v for k, v in train.items()},
               **{("val_loss" if k == "loss" else k): v for k, v in val.items()}}
        with open(run_dir / "metrics.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

        state = {"model": model.state_dict(), "config": cfg, "mean": mean, "std": std, "epoch": epoch,
                 "metrics": val, "vocab": {"triads": TRIADS, "sevenths": SEVENTHS, "extensions": EXTENSIONS}}
        torch.save(state, run_dir / "last.pt")
        if val[metric] > best:
            best, best_epoch, stale = val[metric], epoch, 0
            torch.save(state, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= tc["patience"]:
                print(f"early stop: no {metric} improvement for {stale} epochs")
                break

    print(f"best val {metric} {best:.3f} at epoch {best_epoch}")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    results = {"best_epoch": best_epoch, "val": evaluate(model, val_songs, mean, std, device, cfg),
               "val_untrained_heads": baseline}
    write_predictions(model, val_songs, mean, std, device, cfg, run_dir / "predictions" / "val")
    if test_songs:
        results["test"] = evaluate(model, test_songs, mean, std, device, cfg)
        write_predictions(model, test_songs, mean, std, device, cfg, run_dir / "predictions" / "test")
        print("test: " + " ".join(f"{k} {v:.3f}" for k, v in results["test"].items()))
    if holdout_songs:   # labels from mismatched charts: scored separately, never mixed into test
        results["holdout"] = evaluate(model, holdout_songs, mean, std, device, cfg)
        write_predictions(model, holdout_songs, mean, std, device, cfg, run_dir / "predictions" / "holdout")
        print("holdout: " + " ".join(f"{k} {v:.3f}" for k, v in results["holdout"].items()))
    (run_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
