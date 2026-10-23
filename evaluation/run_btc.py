"""Run stock (pretrained) BTC on audio files and write one .lab per song: the baseline.

Uses the code and checkpoints in third_party/BTC-ISMIR19 without modifying them (via
training/btc_compat.py, which makes that 2019 code run on numpy 2, PyYAML 6 and torch >= 2.6).

    python evaluation/run_btc.py --audio data/songs --out predictions/btc_large
    python evaluation/run_btc.py --audio data/songs --out predictions/btc_majmin --majmin
    python evaluation/evaluate.py --ref labels --est predictions/btc_large
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from btc_compat import (BTC_DIR, BTC_model, HParams, audio_file_to_features,  # noqa: E402
                        idx2chord, idx2voca_chord)

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus")


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_btc(large_voca: bool, device: torch.device):
    config = HParams.load(str(BTC_DIR / "run_config.yaml"))
    if large_voca:
        config.feature["large_voca"] = True
        config.model["num_chords"] = 170
        ckpt, idx_to_chord = BTC_DIR / "test" / "btc_model_large_voca.pt", idx2voca_chord()
    else:
        ckpt, idx_to_chord = BTC_DIR / "test" / "btc_model.pt", idx2chord
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = BTC_model(config=config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, checkpoint["mean"], checkpoint["std"], idx_to_chord


@torch.no_grad()
def predict(audio_path: Path, model, config, mean, std, idx_to_chord, device) -> list[tuple[float, float, str]]:
    cwd = os.getcwd()
    os.chdir(BTC_DIR)            # BTC resolves some paths relative to its own directory
    try:
        feature, time_unit, song_length = audio_file_to_features(str(audio_path), config)
    finally:
        os.chdir(cwd)
    feature = (feature.T - mean) / std
    n = config.model["timestep"]
    n_frames = feature.shape[0]
    feature = np.pad(feature, ((0, (-n_frames) % n), (0, 0)))
    x = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    for t in range(x.shape[1] // n):
        out, _ = model.self_attn_layers(x[:, n * t:n * (t + 1), :])
        p, _ = model.output_layer(out)
        preds.extend(p.squeeze(0).tolist())
    preds = preds[:n_frames]

    segments, start = [], 0
    for i in range(1, len(preds) + 1):
        if i == len(preds) or preds[i] != preds[start]:
            segments.append((start * time_unit, min(i * time_unit, song_length), idx_to_chord[preds[start]]))
            start = i
    return segments


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True, help="audio file or directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--majmin", action="store_true", help="use the 25-class model instead of 170-class")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else default_device()
    files = [args.audio] if args.audio.is_file() else sorted(
        p for p in args.audio.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    model, config, mean, std, idx_to_chord = load_btc(not args.majmin, device)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        segments = predict(f.resolve(), model, config, mean, std, idx_to_chord, device)
        with open(args.out / f"{f.stem}.lab", "w") as fh:
            for s, e, label in segments:
                fh.write(f"{s:.3f}\t{e:.3f}\t{label}\n")
        print(f"{f.stem}: {len(segments)} segments -> {args.out / (f.stem + '.lab')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
