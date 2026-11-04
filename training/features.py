"""BTC input features and frame-level targets.

Features are exactly what pretrained BTC was trained on: log |CQT| at 22.05 kHz, 144 bins,
24 bins per octave, hop 2048, computed in 10 s chunks (108 frames each). One frame is
10/108 s. Raw (un-normalised) features are cached as float16 .npy files; normalisation with
BTC's mean/std happens at training time, after augmentation.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from chord_vocab import EXTENSIONS, IGNORE, Target, encode

SR = 22050
N_BINS = 144
BINS_PER_OCTAVE = 24
BINS_PER_SEMITONE = BINS_PER_OCTAVE // 12
HOP = 2048
CHUNK_SECONDS = 10.0
TIMESTEP = 108
FRAME_SECONDS = CHUNK_SECONDS / TIMESTEP
LOG_FLOOR = float(np.log(1e-6))       # feature value of silence; used to pad pitch shifts


def load_audio(audio_path: Path, sr: int = SR) -> np.ndarray:
    """Mono float32 audio via ffmpeg, so every format works (librosa 1.0 can't read .m4a)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "f32le",
                          "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def compute_features(audio_path: Path) -> np.ndarray:
    """Raw log-CQT, shape [frames, 144]. Matches BTC's audio_file_to_features."""
    import librosa
    y = load_audio(audio_path)
    chunk = int(SR * CHUNK_SECONDS)
    parts, start = [], 0
    while len(y) > start + chunk:
        parts.append(librosa.cqt(y[start:start + chunk], sr=SR, n_bins=N_BINS,
                                 bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP))
        start += chunk
    parts.append(librosa.cqt(y[start:], sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                             hop_length=HOP))
    return np.log(np.abs(np.concatenate(parts, axis=1)) + 1e-6).T.astype(np.float32)


def cached_features(audio_path: Path, cache_dir: Path) -> np.ndarray:
    """Features from cache (keyed by path, size and mtime), computing them if needed."""
    st = audio_path.stat()
    key = hashlib.sha1(f"{audio_path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    path = cache_dir / f"{audio_path.stem}.{key}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    feats = compute_features(audio_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{audio_path.stem}.*.npy"):
        old.unlink()
    np.save(path, feats.astype(np.float16))
    return feats


def read_lab(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append((float(parts[0]), float(parts[1]), parts[2]))
    return out


class FrameTargets:
    """Per-frame targets for one song (numpy arrays, IGNORE where not trainable)."""

    def __init__(self, n: int):
        self.root = np.full(n, IGNORE, np.int64)
        self.triad = np.full(n, IGNORE, np.int64)
        self.seventh = np.full(n, IGNORE, np.int64)
        self.bass = np.full(n, IGNORE, np.int64)
        self.ext = np.zeros((n, len(EXTENSIONS)), np.float32)
        self.ext_mask = np.zeros(n, np.float32)
        self.labels = np.full(n, "X", dtype=object)    # Harte label per frame, for metrics

    def __len__(self):
        return len(self.root)

    def set(self, sl: slice, t: Target, label: str):
        self.root[sl], self.triad[sl], self.seventh[sl], self.bass[sl] = t.root, t.triad, t.seventh, t.bass
        if t.ext is not None:
            self.ext[sl] = t.ext
            self.ext_mask[sl] = 1.0
        self.labels[sl] = label


def frame_targets(segments: list[tuple[float, float, str]], n_frames: int) -> FrameTargets:
    """Label each frame by the segment containing its centre; frames outside any segment are X."""
    ft = FrameTargets(n_frames)
    for start, end, label in segments:
        a = max(int(np.ceil(start / FRAME_SECONDS - 0.5)), 0)
        b = min(int(np.ceil(end / FRAME_SECONDS - 0.5)), n_frames)
        if b > a:
            ft.set(slice(a, b), encode(label), label)
    return ft


def pitch_shift(feats: np.ndarray, semitones: int) -> np.ndarray:
    """Shift log-CQT frames [T, bins] by whole semitones (2 bins each), padding with silence."""
    if semitones == 0:
        return feats
    k = semitones * BINS_PER_SEMITONE
    out = np.full_like(feats, LOG_FLOOR)
    if k > 0:
        out[:, k:] = feats[:, :-k]
    else:
        out[:, :k] = feats[:, -k:]
    return out
