"""Factorised chord vocabulary: Harte label <-> per-head training targets.

A chord is split into five independent targets, all relative to the root except the root itself,
so pitch-shift augmentation only has to move the root:

    root       13 classes   C .. B, N
    triad       7 classes   maj min dim aug sus2 sus4 5
    seventh     5 classes   none b7 7 6 bb7
    extensions  8 labels    b5 9 b9 #9 11 #11 13 b13   (multi-label)
    bass       12 classes   interval of the bass above the root (0 = root position)

So "G:maj(9)/3" (Gadd9/B) is root=G, triad=maj, seventh=none, ext={9}, bass=4, and rare
combinations never become empty softmax classes.

N frames train only the root head (root=N); X frames train nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "labeling"))
from chord_utils import BASS_INTERVALS, SHARP_NAMES, degree_semitones, parse_harte  # noqa: E402

IGNORE = -100

ROOTS = SHARP_NAMES + ["N"]
N_ROOT = 12
TRIADS = ["maj", "min", "dim", "aug", "sus2", "sus4", "5"]
SEVENTHS = ["none", "b7", "7", "6", "bb7"]
EXTENSIONS = ["b5", "9", "b9", "#9", "11", "#11", "13", "b13"]
N_BASS = 12

HEAD_SIZES = {"root": len(ROOTS), "triad": len(TRIADS), "seventh": len(SEVENTHS),
              "ext": len(EXTENSIONS), "bass": N_BASS}

# (triad, seventh) -> Harte shorthand; other combinations render the seventh as an extension.
_BASE = {
    ("maj", "none"): "maj", ("min", "none"): "min", ("dim", "none"): "dim",
    ("aug", "none"): "aug", ("sus2", "none"): "sus2", ("sus4", "none"): "sus4",
    ("5", "none"): "5",
    ("maj", "7"): "maj7", ("maj", "b7"): "7", ("min", "b7"): "min7", ("min", "7"): "minmaj7",
    ("dim", "b7"): "hdim7", ("dim", "bb7"): "dim7", ("maj", "6"): "maj6", ("min", "6"): "min6",
}
# Stacking extensions onto a shorthand: (shorthand, extension) -> bigger shorthand.
_UPGRADE = {("7", "9"): "9", ("maj7", "9"): "maj9", ("min7", "9"): "min9",
            ("9", "11"): "11", ("min9", "11"): "min11",
            ("11", "13"): "13", ("min11", "13"): "min13"}


class Target(tuple):
    """(root, triad, seventh, ext, bass); ext is a tuple of 0/1 or None when ignored."""

    __slots__ = ()

    def __new__(cls, root, triad, seventh, ext, bass):
        return super().__new__(cls, (root, triad, seventh, ext, bass))

    root = property(lambda s: s[0])
    triad = property(lambda s: s[1])
    seventh = property(lambda s: s[2])
    ext = property(lambda s: s[3])
    bass = property(lambda s: s[4])


IGNORE_TARGET = Target(IGNORE, IGNORE, IGNORE, None, IGNORE)
NO_CHORD_TARGET = Target(N_ROOT, IGNORE, IGNORE, None, IGNORE)


def encode(label: str) -> Target:
    """Harte label -> Target. Unknown / unparseable labels are ignored (like X)."""
    label = label.strip()
    if label == "N":
        return NO_CHORD_TARGET
    if label == "X":
        return IGNORE_TARGET
    try:
        root, degrees, bass = parse_harte(label)
    except (ValueError, KeyError):
        return IGNORE_TARGET
    d = set(degrees)

    if "3" in d:
        triad = "aug" if "#5" in d and "5" not in d else "maj"
    elif "b3" in d:
        triad = "dim" if "b5" in d and "5" not in d else "min"
    elif "4" in d:
        triad = "sus4"
    elif "2" in d:
        triad = "sus2"
    else:
        triad = "5"

    if "bb7" in d:
        seventh = "bb7"
    elif "b7" in d:
        seventh = "b7"
    elif "7" in d:
        seventh = "7"
    elif "6" in d:
        seventh = "6"
    else:
        seventh = "none"

    ext = set()
    if "b5" in d and triad in ("maj", "min") and "5" not in d:
        ext.add("b5")
    if "9" in d or ("2" in d and triad != "sus2"):
        ext.add("9")
    if "11" in d or ("4" in d and triad != "sus4"):
        ext.add("11")
    if "#11" in d or "#4" in d:
        ext.add("#11")
    if "b6" in d:
        ext.add("b13")
    ext |= d & {"b9", "#9", "13", "b13"}

    bass_iv = degree_semitones(bass) % 12 if bass else 0
    return Target(root, TRIADS.index(triad), SEVENTHS.index(seventh),
                  tuple(int(e in ext) for e in EXTENSIONS), bass_iv)


def decode(root: int, triad: int, seventh: int, ext, bass: int) -> str:
    """Head predictions (class indices; ext as 0/1 sequence) -> Harte label."""
    if root == N_ROOT:
        return "N"
    t, s = TRIADS[triad], SEVENTHS[seventh]
    exts = [e for e, on in zip(EXTENSIONS, ext) if on]
    extras: list[str] = []
    if (t, s) in _BASE:
        base = _BASE[(t, s)]
    else:
        base = _BASE[(t, "none")]
        extras.append(s)
    for e in ("9", "11", "13"):
        if e in exts and (base, e) in _UPGRADE:
            base = _UPGRADE[(base, e)]
            exts.remove(e)
    if base == "maj9" and {"11", "13"} <= set(exts):
        base = "maj13"
        exts = [e for e in exts if e not in ("11", "13")]
    if "b5" in exts:
        exts.remove("b5")
        extras += ["*5", "b5"]
    extras += exts
    label = f"{SHARP_NAMES[root]}:{base}"
    if extras:
        label += "(" + ",".join(extras) + ")"
    if bass:
        name = BASS_INTERVALS[bass]
        if bass == 6 and t == "dim":
            name = "b5"
        elif bass == 8 and t == "aug":
            name = "#5"
        label += "/" + name
    return label


def transpose_target(t: Target, semitones: int) -> Target:
    if t.root in (IGNORE, N_ROOT):
        return t
    return Target((t.root + semitones) % 12, t.triad, t.seventh, t.ext, t.bass)


def _self_test() -> None:
    import mir_eval
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "labeling"))
    from chord_utils import _HARTE_CASES

    def same(a: str, b: str) -> bool:
        ra, ba, sa = mir_eval.chord.encode(a)
        rb, bb, sb = mir_eval.chord.encode(b)
        return ra == rb and sa == sb and (ba == bb).all()

    for sym, label in _HARTE_CASES.items():
        t = encode(label)
        back = decode(t.root, t.triad, t.seventh, t.ext or [0] * 8, t.bass) if t.root != IGNORE else label
        if label == "N":
            assert back == "N"
            continue
        # Altered-13 / natural-11 subtleties aside, the pitch content must survive the round trip.
        assert same(label, back) or label in ("G:13(*9,b9)",), f"{sym}: {label} -> {t} -> {back}"
    assert encode("X") == IGNORE_TARGET and encode("rubbish") == IGNORE_TARGET
    g = encode("G:maj(9)/3")
    assert (g.root, TRIADS[g.triad], SEVENTHS[g.seventh], g.bass) == (7, "maj", "none", 4)
    assert transpose_target(g, 2).root == 9 and decode(*transpose_target(g, 2)) == "A:maj(9)/3"
    assert decode(*encode("C:13")) == "C:13" and decode(*encode("E:sus4(b7)")) == "E:sus4(b7)"
    print(f"chord_vocab: {len(_HARTE_CASES)} labels round-trip OK")


if __name__ == "__main__":
    _self_test()
