"""Run a fine-tuned checkpoint on audio and write .lab files.

    python training/predict.py --ckpt runs/finetune/best.pt --audio song.m4a --out predictions/ft
    python training/predict.py --ckpt runs/finetune/best.pt --audio data/songs --out predictions/ft
    python evaluation/evaluate.py --ref labels --est predictions/ft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from dataset import AUDIO_EXTS  # noqa: E402
from features import FRAME_SECONDS, cached_features, compute_features  # noqa: E402
from inference import decode_frames, frames_to_segments, predict_probs  # noqa: E402
from model import FactorisedBTC  # noqa: E402
from train import pick_device, resolve  # noqa: E402


def load_model(path: Path, device) -> tuple[FactorisedBTC, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = ckpt["config"]["model"]
    model = FactorisedBTC(head_dropout=m["head_dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True, help="audio file or directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ext-threshold", type=float, default=None)
    ap.add_argument("--min-duration", type=float, default=None, help="merge segments shorter than this (s)")
    ap.add_argument("--cache", type=Path, default=None, help="feature cache dir (default: no cache)")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    model, ckpt = load_model(args.ckpt, device)
    pc = ckpt["config"].get("predict", {})
    thr = args.ext_threshold if args.ext_threshold is not None else pc.get("ext_threshold", 0.5)
    min_dur = args.min_duration if args.min_duration is not None else pc.get("min_duration", 0.0)

    files = [args.audio] if args.audio.is_file() else sorted(
        p for p in args.audio.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        feats = cached_features(f, resolve(args.cache)) if args.cache else compute_features(f)
        labels = decode_frames(predict_probs(model, feats, ckpt["mean"], ckpt["std"], device), thr)
        segs = frames_to_segments(labels, len(feats) * FRAME_SECONDS, min_dur)
        with open(args.out / f"{f.stem}.lab", "w") as fh:
            for a, b, lab in segs:
                fh.write(f"{a:.3f}\t{b:.3f}\t{lab}\n")
        print(f"{f.stem}: {len(segs)} segments -> {args.out / (f.stem + '.lab')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
