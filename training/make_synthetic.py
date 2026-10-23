"""Generate synthetic songs (audio + .lab) with worship/gospel-style chords.

For testing the training code before real data exists, and as a sanity check that the model
can learn slash chords and extensions at all. Not a substitute for real recordings.

    python training/make_synthetic.py --out cache/synthetic --songs 24 --seconds 60
    python training/train.py --set "data.sources=[{audio: cache/synthetic/audio, labels: cache/synthetic/labels}]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "labeling"))
from chord_utils import (NOTE_TO_PC, chord_to_harte, degree_semitones, parse_harte,  # noqa: E402
                         transpose_symbol)

SR = 22050

# Progressions in G; each song is transposed to a random key.
PROGRESSIONS = [
    ["G", "D/F#", "Em7", "Cadd9"],
    ["Cadd9", "G/B", "Am7", "Dsus4", "D"],
    ["Em7", "Cmaj7", "G", "D/F#"],
    ["G", "G/B", "C", "D", "Em", "D/F#", "C", "Dsus4"],
    ["Cmaj9", "Bm7", "Em9", "Am11", "D13"],
    ["G13", "Cmaj7", "F#m7b5", "B7#9", "Em9", "A9", "Am7", "D7sus4"],
    ["Em7", "A7sus4", "Dmaj7", "Gmaj9", "C#m7b5", "F#7b9", "Bm9"],
    ["G", "Am/G", "G/B", "C6/9", "D/C", "Bm7", "E7#9", "Am9", "D9sus"],
    ["Cadd9", "D", "Em", "G/D", "C", "D/C", "Bm7", "Em"],
    ["G", "Bbdim7", "Am7", "D7", "Gmaj7", "G7", "Cmaj7", "Cm6"],
]


def tone(freq: float, dur: float, amp: float, rng) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-t * rng.uniform(0.8, 2.0)) * np.minimum(t / 0.01, 1.0)
    sig = sum(np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 6.28)) / h ** 1.3 for h in range(1, 6))
    return amp * env * sig


def voice(label: str) -> tuple[int, list[int]]:
    """Harte label -> (bass MIDI note, chord MIDI notes)."""
    root, degrees, bass = parse_harte(label)
    bass_pc = (root + degree_semitones(bass)) % 12 if bass else root
    bass_note = 36 + bass_pc
    notes, base = [], 48 + root
    for d in degrees:
        n = base + degree_semitones(d)
        while n > 76:
            n -= 12
        notes.append(n)
    return bass_note, sorted(set(notes))


def make_song(rng, seconds: float):
    prog = PROGRESSIONS[rng.integers(len(PROGRESSIONS))]
    shift = int(rng.integers(-5, 7))
    prog = [transpose_symbol(c, shift) for c in prog]
    bpm = rng.uniform(65, 130)
    beat = 60.0 / bpm
    audio = np.zeros(int(seconds * SR) + SR)
    segs = []
    t = rng.uniform(0.5, 2.0)
    segs.append((0.0, t, "N"))
    i = 0
    while t < seconds - 2.0:
        sym = prog[i % len(prog)]
        label = chord_to_harte(sym)
        dur = beat * int(rng.choice([2, 4, 4, 8]))
        dur = min(dur, seconds - 1.0 - t)
        bass, notes = voice(label)
        s = int(t * SR)
        for k in range(int(np.ceil(dur / beat))):      # re-strike each beat, like a strummed pad
            off = s + int(k * beat * SR)
            length = min(beat * 1.5, dur - k * beat + 0.05)
            seg = tone(440 * 2 ** ((bass - 69) / 12), length, 0.35, rng)
            for n in notes:
                seg = seg + tone(440 * 2 ** ((n - 69) / 12), length, 0.18, rng)
            audio[off:off + len(seg)] += seg[: len(audio) - off]
        segs.append((t, t + dur, label))
        t += dur
        i += 1
    segs.append((t, seconds, "N"))
    audio = audio[: int(seconds * SR)]
    audio += rng.normal(0, 0.003, len(audio))
    return audio / (np.abs(audio).max() + 1e-9) * 0.8, segs


def main(argv=None) -> int:
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("cache/synthetic"))
    ap.add_argument("--songs", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    for k in range(args.songs):
        audio, segs = make_song(rng, args.seconds)
        sid = f"synth_{k:03d}"
        sf.write(args.out / "audio" / f"{sid}.wav", audio, SR)
        with open(args.out / "labels" / f"{sid}.lab", "w") as f:
            for a, b, lab in segs:
                f.write(f"{a:.3f}\t{b:.3f}\t{lab}\n")
    print(f"wrote {args.songs} songs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
