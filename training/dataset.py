"""Songs -> training windows, with pitch-shift and gain augmentation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from chord_vocab import IGNORE, N_ROOT
from features import FrameTargets, LOG_FLOOR, cached_features, frame_targets, pitch_shift, read_lab

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus")


@dataclass
class Song:
    song_id: str
    audio: Path
    lab: Path
    feats: np.ndarray | None = None        # raw log-CQT [T, 144]
    targets: FrameTargets | None = None

    def load(self, cache_dir: Path) -> "Song":
        if self.feats is None:
            self.feats = cached_features(self.audio, cache_dir)
            self.targets = frame_targets(read_lab(self.lab), len(self.feats))
        return self


def discover_songs(sources: list[dict]) -> dict[str, Song]:
    """Pair audio and .lab files by song id. sources: [{audio: dir, labels: dir}, ...].

    Labels may be <id>.lab or <id>/chords.lab (pipeline output layout).
    """
    songs: dict[str, Song] = {}
    for src in sources:
        audio_dir, label_dir = Path(src["audio"]), Path(src["labels"])
        if not audio_dir.exists() or not label_dir.exists():
            continue
        labs = {p.stem: p for p in label_dir.glob("*.lab")}
        labs.update({p.parent.name: p for p in label_dir.glob("*/chords.lab")})
        for a in sorted(audio_dir.iterdir()):
            if a.suffix.lower() in AUDIO_EXTS and a.stem in labs and a.stem not in songs:
                songs[a.stem] = Song(a.stem, a, labs[a.stem])
    return songs


def song_group(song_id: str) -> str:
    """Versions of one song share a group: 'goodness_of_god__live' -> 'goodness_of_god'."""
    return song_id.split("__")[0]


def make_splits(song_ids: list[str], val: float, test: float, seed: int) -> dict[str, list[str]]:
    """Split by song group (never by segment), deterministic for a given seed."""
    groups = sorted({song_group(s) for s in song_ids})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test = max(1, round(n * test)) if test > 0 and n >= 3 else 0
    n_val = max(1, round(n * val)) if val > 0 and n - n_test >= 2 else 0
    test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    out = {"train": [], "val": [], "test": []}
    for s in sorted(song_ids):
        g = song_group(s)
        out["test" if g in test_g else "val" if g in val_g else "train"].append(s)
    return out


def load_or_create_splits(path: Path, song_ids: list[str], val: float, test: float, seed: int):
    """Existing split file wins (new songs go to train); otherwise create and save one."""
    if path.exists():
        splits = json.loads(path.read_text())
        known = {s for v in splits.values() for s in v}
        new = [s for s in song_ids if s not in known]
        if new:
            # Keep new versions of already-split songs with their group.
            for s in new:
                where = next((k for k, v in splits.items() if any(song_group(x) == song_group(s) for x in v)),
                             "train")
                splits[where].append(s)
            path.write_text(json.dumps(splits, indent=2))
        return {k: [s for s in v if s in set(song_ids)] for k, v in splits.items()}
    splits = make_splits(song_ids, val, test, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))
    return splits


def _window(song: Song, start: int, length: int, shift: int, gain: float):
    t = song.targets
    end = min(start + length, len(t))
    n = end - start
    x = np.full((length, song.feats.shape[1]), LOG_FLOOR, np.float32)
    x[:n] = pitch_shift(song.feats[start:end], shift) + gain
    root = np.full(length, IGNORE, np.int64)
    root[:n] = t.root[start:end]
    if shift:
        r = root[:n]
        chord = (r != IGNORE) & (r != N_ROOT)
        r[chord] = (r[chord] + shift) % 12
    out = {"x": x, "root": root}
    for name in ("triad", "seventh", "bass"):
        a = np.full(length, IGNORE, np.int64)
        a[:n] = getattr(t, name)[start:end]
        out[name] = a
    ext = np.zeros((length, t.ext.shape[1]), np.float32)
    ext[:n] = t.ext[start:end]
    mask = np.zeros(length, np.float32)
    mask[:n] = t.ext_mask[start:end]
    out.update(ext=ext, ext_mask=mask, valid=np.arange(length) < n)
    return out


class TrainWindows(Dataset):
    """Random windows from training songs, each with a random pitch shift and gain."""

    def __init__(self, songs: list[Song], mean: float, std: float, timestep: int = 108,
                 windows_per_epoch: int | None = None, max_shift: int = 5, gain_db: float = 6.0):
        self.songs, self.mean, self.std, self.timestep = songs, mean, std, timestep
        self.max_shift, self.gain = max_shift, gain_db * np.log(10) / 20     # dB -> natural log
        lengths = np.array([len(s.feats) for s in songs], dtype=float)
        self.weights = lengths / lengths.sum()
        self.n = windows_per_epoch or max(int(lengths.sum() / timestep) * 4, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        song = self.songs[rng.choice(len(self.songs), p=self.weights)]
        start = int(rng.integers(0, max(len(song.feats) - self.timestep, 0) + 1))
        shift = int(rng.integers(-self.max_shift, self.max_shift + 1)) if self.max_shift else 0
        gain = float(rng.uniform(-self.gain, self.gain)) if self.gain else 0.0
        w = _window(song, start, self.timestep, shift, gain)
        w["x"] = (w["x"] - self.mean) / self.std
        return {k: torch.from_numpy(np.asarray(v)) for k, v in w.items()}


class EvalWindows(Dataset):
    """Consecutive, non-overlapping windows covering every song exactly once (no augmentation)."""

    def __init__(self, songs: list[Song], mean: float, std: float, timestep: int = 108):
        self.songs, self.mean, self.std, self.timestep = songs, mean, std, timestep
        self.index = [(si, start) for si, s in enumerate(songs)
                      for start in range(0, len(s.feats), timestep)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, start = self.index[i]
        w = _window(self.songs[si], start, self.timestep, 0, 0.0)
        w["x"] = (w["x"] - self.mean) / self.std
        item = {k: torch.from_numpy(np.asarray(v)) for k, v in w.items()}
        item["song"], item["start"] = si, start
        return item
