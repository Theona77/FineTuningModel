"""Chord parsing, Harte conversion and lyric-to-chord alignment.

Pure Python (standard library only) so it can be tested without a GPU or any
audio packages:

    python chord_utils.py          # runs the self-tests

Main entry points
-----------------
chord_to_harte(symbol)            chart symbol ("G/B", "F9sus") -> Harte ("G:maj/3")
transpose_symbol(symbol, n)       transpose a chart symbol by n semitones
transpose_harte(label, n)         transpose a Harte label by n semitones
harte_pitch_classes(label)        set of pitch classes (0 = C) sounding in a label
parse_chart(text)                 ChordPro or chords-over-lyrics -> list[ChartLine]
align_chart(lines, words, ...)    chart + timed words + beats -> AlignmentResult
write_lab(path, segments)         HTK-style .lab file
"""

from __future__ import annotations

import bisect
import csv
import difflib
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------

NOTE_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Semitone -> Harte interval name, used for slash-chord basses.
BASS_INTERVALS = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]

# Harte degree -> semitones above the root.
_DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 9: 14, 11: 17, 13: 21}

# Harte shorthands (the mir_eval set) -> degrees they contain.
SHORTHANDS = {
    "maj": ["1", "3", "5"],
    "min": ["1", "b3", "5"],
    "dim": ["1", "b3", "b5"],
    "aug": ["1", "3", "#5"],
    "maj7": ["1", "3", "5", "7"],
    "min7": ["1", "b3", "5", "b7"],
    "7": ["1", "3", "5", "b7"],
    "dim7": ["1", "b3", "b5", "bb7"],
    "hdim7": ["1", "b3", "b5", "b7"],
    "minmaj7": ["1", "b3", "5", "7"],
    "maj6": ["1", "3", "5", "6"],
    "min6": ["1", "b3", "5", "6"],
    "9": ["1", "3", "5", "b7", "9"],
    "maj9": ["1", "3", "5", "7", "9"],
    "min9": ["1", "b3", "5", "b7", "9"],
    "11": ["1", "3", "5", "b7", "9", "11"],
    "min11": ["1", "b3", "5", "b7", "9", "11"],
    "13": ["1", "3", "5", "b7", "9", "11", "13"],
    "maj13": ["1", "3", "5", "7", "9", "11", "13"],
    "min13": ["1", "b3", "5", "b7", "9", "11", "13"],
    "sus2": ["1", "2", "5"],
    "sus4": ["1", "4", "5"],
    "1": ["1"],
    "5": ["1", "5"],
}


def degree_semitones(degree: str) -> int:
    """'b7' -> 10, '#11' -> 18, 'bb7' -> 9."""
    m = re.fullmatch(r"([b#]*)(\d+)", degree)
    if not m:
        raise ValueError(f"bad degree {degree!r}")
    acc, num = m.groups()
    return _DEGREE_SEMITONES[int(num)] - acc.count("b") + acc.count("#")


def _degree_sort_key(degree: str):
    m = re.fullmatch(r"\*?([b#]*)(\d+)", degree)
    return (int(m.group(2)), m.group(1)) if m else (99, degree)


def _note_name(pc: int, flats: bool) -> str:
    return (FLAT_NAMES if flats else SHARP_NAMES)[pc % 12]


def _normalise_accidentals(s: str) -> str:
    return s.replace("♯", "#").replace("♭", "b")


# ---------------------------------------------------------------------------
# Chord symbol -> Harte
# ---------------------------------------------------------------------------

_ROOT_RE = re.compile(r"^([A-G])([#b]?)(.*)$")
_NOTE_RE = re.compile(r"^[A-G][#b]?$")
NO_CHORD = {"N.C.", "N.C", "NC", "N", "N/C", "NO CHORD"}


def _split_symbol(symbol: str):
    """Return (root, quality, bass) for a chart symbol, or raise ValueError."""
    s = _normalise_accidentals(symbol.strip())
    m = _ROOT_RE.match(s)
    if not m:
        raise ValueError(f"not a chord: {symbol!r}")
    root, rest = m.group(1) + m.group(2), m.group(3)
    bass = None
    if "/" in rest:
        head, tail = rest.rsplit("/", 1)
        tail = _normalise_accidentals(tail.strip())
        if _NOTE_RE.match(tail):
            rest, bass = head, tail
        elif tail == "9" and head.endswith("6"):       # C6/9
            rest = head + "9"
        else:
            raise ValueError(f"bad slash in {symbol!r}")
    return root, rest, bass


def _normalise_quality(q: str) -> str:
    q = q.replace("(", "").replace(")", "").replace(",", "").replace(" ", "")
    q = q.replace("Δ", "M").replace("△", "M").replace("°", "dim")
    q = re.sub(r"^ø7?", "m7b5", q)
    q = re.sub(r"^o(?=7|$)", "dim", q)
    q = re.sub(r"^(min|mi|-)", "m", q)
    # Upper-case M means major seventh; must be handled before lower-casing.
    q = re.sub(r"^(m?)(?:Maj|MAJ|Ma|M)", r"\1maj", q)
    return q.lower()


@dataclass
class _Quality:
    third: str = "maj"          # maj | min | dim | aug | power
    seventh: str | None = None  # None | 'b7' | '7' | 'bb7'
    ext: int = 0                # 0 | 9 | 11 | 13 (stacked extension)
    sixth: bool = False
    sus: str | None = None      # None | '2' | '4'
    adds: list = field(default_factory=list)
    alts: list = field(default_factory=list)


def _parse_quality(q: str, symbol: str) -> _Quality:
    r = _Quality()
    if q.startswith("hdim"):
        r.third, r.seventh, q = "dim", "b7", q[4:].lstrip("7")
    elif q.startswith("dim"):
        r.third, q = "dim", q[3:]
        if q.startswith("7"):
            r.seventh, q = "bb7", q[1:]
    elif q.startswith("aug"):
        r.third, q = "aug", q[3:]
    elif q.startswith("+"):
        r.third, q = "aug", q[1:]
    elif q.startswith("m") and not q.startswith("maj"):
        r.third, q = "min", q[1:]

    major_seventh = False
    if q.startswith("maj"):
        major_seventh, q = True, q[3:]

    m = re.match(r"^(13|11|9|7|69|6|5|4|2)", q)
    if m:
        num, q = m.group(1), q[m.end():]
        if num in ("7", "9", "11", "13"):
            r.seventh = r.seventh or "b7"
            if num != "7":
                r.ext = int(num)
        elif num == "69":
            r.sixth = True
            r.adds.append("9")
        elif num == "6":
            r.sixth = True
        elif num == "5":
            if r.third == "maj" and not major_seventh:
                r.third = "power"
            else:
                r.alts.append("b5")    # unusual, but keep the information
        elif num == "2":
            r.sus = "2"
        elif num == "4":
            r.sus = "4"
    if major_seventh:
        if r.seventh or r.third == "min":
            r.seventh = "7"
        elif not r.sixth:
            pass                        # "Cmaj" = plain major triad

    while q:
        m = re.match(r"^sus(2|4)?", q)
        if m:
            r.sus, q = m.group(1) or "4", q[m.end():]
            continue
        m = re.match(r"^add([b#]?)(2|4|6|9|11|13)", q)
        if m:
            acc, deg = m.groups()
            deg = {"2": "9", "4": "11", "6": "13"}[deg] if deg in ("2", "4", "6") else deg
            if deg == "13" and not acc and m.group(2) == "6":
                r.sixth = True
            else:
                r.adds.append(acc + deg)
            q = q[m.end():]
            continue
        m = re.match(r"^([b#])(5|9|11|13)", q)
        if m:
            r.alts.append(m.group(1) + m.group(2))
            q = q[m.end():]
            continue
        if q.startswith("alt"):
            r.seventh = r.seventh or "b7"
            r.alts += ["b9", "#9", "b13"]
            q = q[3:]
            continue
        m = re.match(r"^(9|11|13)", q)          # e.g. "Gsus9", "C7/13" written "C713"
        if m:
            r.adds.append(m.group(1))
            q = q[m.end():]
            continue
        raise ValueError(f"can't parse quality {q!r} in {symbol!r}")

    if r.third == "min" and r.seventh == "b7" and "b5" in r.alts:
        r.third = "dim"                         # m7b5 -> hdim7
        r.alts.remove("b5")
    if r.third == "maj" and "#5" in r.alts and r.seventh in (None, "b7") and not r.sus:
        r.third = "aug"                         # 7#5 / C(#5) -> aug
        r.alts.remove("#5")
    return r


def _render_quality(r: _Quality) -> str:
    extras: list[str] = []
    if r.third == "power":
        return "(1,5)"

    if r.sus:
        base = "sus" + r.sus
        if r.seventh:
            extras.append(r.seventh)
        if r.ext >= 9:
            extras.append("9")
        if r.ext == 13:
            extras.append("13")
        if r.sixth:
            extras.append("6")
    elif r.third == "aug":
        base = "aug"
        if r.seventh:
            extras.append(r.seventh)
        extras += [str(e) for e in (9, 11, 13) if r.ext >= e and (e == 9 or e == r.ext)]
    elif r.third == "dim":
        base = {"bb7": "dim7", "b7": "hdim7"}.get(r.seventh or "", "dim")
        extras += [str(e) for e in (9, 11, 13) if r.ext >= e and (e == 9 or e == r.ext)]
    elif r.third == "min":
        if r.seventh == "b7":
            base = {0: "min7", 9: "min9", 11: "min11", 13: "min13"}[r.ext]
        elif r.seventh == "7":
            base = "minmaj7"
            extras += [str(e) for e in (9, 11, 13) if r.ext >= e and (e == 9 or e == r.ext)]
        elif r.sixth:
            base = "min6"
        else:
            base = "min"
    else:
        if r.seventh == "b7":
            base = {0: "7", 9: "9", 11: "11", 13: "13"}[r.ext]
        elif r.seventh == "7":
            base = {0: "maj7", 9: "maj9", 11: "maj9", 13: "maj13"}[r.ext]
            if r.ext == 11:
                extras.append("11")
        elif r.sixth:
            base = "maj6"
        else:
            base = "maj"
        if r.sixth and r.seventh:
            extras.append("13")

    implied = set(SHORTHANDS[base]) | set(extras)
    for a in r.adds:
        if a not in implied:
            extras.append(a)
            implied.add(a)
    for a in r.alts:
        natural = a.lstrip("b#")
        if natural in implied:
            extras.append("*" + natural)
        if a not in implied:
            extras.append(a)
            implied.add(a)
    extras = sorted(dict.fromkeys(extras), key=_degree_sort_key)
    return f"{base}({','.join(extras)})" if extras else base


def chord_to_harte(symbol: str) -> str:
    """Convert a chart chord symbol to a Harte label. Raises ValueError if unparseable."""
    s = symbol.strip()
    if s.upper() in NO_CHORD:
        return "N"
    root, q, bass = _split_symbol(s)
    r = _parse_quality(_normalise_quality(q), symbol)
    label = f"{root}:{_render_quality(r)}"
    if bass:
        interval = (NOTE_TO_PC[bass] - NOTE_TO_PC[root]) % 12
        if interval:
            name = BASS_INTERVALS[interval]
            if interval == 6 and r.third == "dim":
                name = "b5"
            elif interval == 8 and r.third == "aug":
                name = "#5"
            label += "/" + name
    return label


def is_chord(token: str) -> bool:
    try:
        chord_to_harte(token)
        return True
    except (ValueError, KeyError):
        return False


def transpose_symbol(symbol: str, semitones: int) -> str:
    """Transpose a chart symbol, keeping its quality text as written."""
    if semitones % 12 == 0 or symbol.strip().upper() in NO_CHORD:
        return symbol
    root, q, bass = _split_symbol(symbol)
    flats = "b" in root or (bass is not None and "b" in bass)
    out = _note_name(NOTE_TO_PC[root] + semitones, flats) + q
    if bass:
        out += "/" + _note_name(NOTE_TO_PC[bass] + semitones, flats)
    return out


def transpose_harte(label: str, semitones: int) -> str:
    """Transpose a Harte label (bass intervals are relative, so only the root moves)."""
    if label in ("N", "X") or semitones % 12 == 0:
        return label
    root, rest = label.split(":", 1) if ":" in label else (label, "maj")
    return f"{_note_name(NOTE_TO_PC[root] + semitones, 'b' in root)}:{rest}"


def parse_harte(label: str):
    """Return (root_pc, degrees, bass_degree) for a Harte label; (None, [], None) for N/X."""
    if label in ("N", "X"):
        return None, [], None
    m = re.fullmatch(r"([A-G][#b]*)(?::([^(/]*)(?:\(([^)]*)\))?)?(?:/([b#]*\d+))?", label)
    if not m:
        raise ValueError(f"bad Harte label {label!r}")
    root, short, ext, bass = m.groups()
    if root not in NOTE_TO_PC:
        raise ValueError(f"bad root in {label!r}")
    if short is None:
        short = "maj"
    if short == "" and not ext:
        raise ValueError(f"empty quality in {label!r}")
    degrees = list(SHORTHANDS[short]) if short else []
    for e in (ext.split(",") if ext else []):
        e = e.strip()
        if e.startswith("*"):
            if e[1:] in degrees:
                degrees.remove(e[1:])
        elif e not in degrees:
            degree_semitones(e)
            degrees.append(e)
    if bass:
        degree_semitones(bass)
    return NOTE_TO_PC[root], degrees, bass


def harte_pitch_classes(label: str) -> set[int]:
    root, degrees, bass = parse_harte(label)
    if root is None:
        return set()
    pcs = {(root + degree_semitones(d)) % 12 for d in degrees}
    if bass:
        pcs.add((root + degree_semitones(bass)) % 12)
    return pcs


# ---------------------------------------------------------------------------
# Chart parsing
# ---------------------------------------------------------------------------

@dataclass
class ChordAnchor:
    symbol: str
    harte: str
    word_index: int | None     # index into ChartLine.words; None for chord-only lines
    trailing: bool = False     # sits after the last word -> anchored to word end
    beat_offset: int = 0       # 2nd, 3rd... trailing chord: this many beats after the word end


@dataclass
class ChartLine:
    index: int
    section: str
    words: list[str]
    chords: list[ChordAnchor]
    text: str
    stacked: list[str] = field(default_factory=list)   # chords dropped by STACKED_CHORDS

    @property
    def is_instrumental(self) -> bool:
        return not self.words and bool(self.chords)

    def chord_key(self) -> tuple:
        return tuple((c.harte, c.word_index, c.trailing, c.beat_offset) for c in self.chords)


_SECTION_RE = re.compile(
    r"^\s*[\[(]?\s*(intro|verse|chorus|pre[- ]?chorus|bridge|tag|outro|interlude|instrumental|"
    r"vamp|ending|refrain|turnaround|hook|coda|break|channel|post[- ]?chorus)\b[^\]\)]*[\])]?\s*:?\s*$",
    re.IGNORECASE,
)
_DIRECTIVE_RE = re.compile(r"^\s*\{\s*([^:}]+?)\s*(?::\s*(.*?))?\s*\}\s*$")
_CHORD_LINE_JUNK = re.compile(r"^(\||-+|/|\.|x\d+|\(x\d+\)|\d+x|\(|\)|%|:\|\||\|\|:)$", re.IGNORECASE)
_REPEAT_TOKEN = re.compile(r"^\(?(x\d+|\d+x|repeat)\)?$", re.IGNORECASE)
_REPEAT_LINE = re.compile(r"^\s*[\[(]?\s*(repeat|play)\b.*$", re.IGNORECASE)   # "REPEAT CHORUS"


def _bracket_chords(content: str) -> list[str]:
    """Chords inside one [...]: a single chord, or bar notation like "| Ab / / Db | Ab / / / |"."""
    if is_chord(content):
        return [content]
    tokens = content.split()
    chords = [t for t in tokens if is_chord(t)]
    if chords and all(is_chord(t) or _CHORD_LINE_JUNK.match(t) for t in tokens):
        return chords
    return []


def _is_chord_line(line: str) -> bool:
    tokens = line.split()
    chords = 0
    for t in tokens:
        t = t.strip("()|")
        if not t or _CHORD_LINE_JUNK.match(t):
            continue
        if not is_chord(t):
            return False
        chords += 1
    return chords > 0


def chords_over_lyrics_to_chordpro(text: str) -> str:
    """Convert a chords-over-lyrics chart (chords on their own line above lyrics) to ChordPro."""
    out: list[str] = []
    lines = text.expandtabs(4).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            out.append("")
            i += 1
            continue
        if _SECTION_RE.match(line):
            out.append("{comment: %s}" % line.strip().strip("[]():").strip())
            i += 1
            continue
        if _is_chord_line(line):
            chords = [(m.start(), m.group()) for m in re.finditer(r"\S+", line)
                      if is_chord(m.group().strip("()|"))]
            chords = [(p, c.strip("()|")) for p, c in chords]
            nxt = lines[i + 1].rstrip() if i + 1 < len(lines) else ""
            if nxt.strip() and not _is_chord_line(nxt) and not _SECTION_RE.match(nxt):
                lyric = nxt.ljust(max(p for p, _ in chords) + 1)
                for pos, c in sorted(chords, reverse=True):
                    lyric = lyric[:pos] + f"[{c}]" + lyric[pos:]
                out.append(lyric.rstrip())
                i += 2
            else:
                out.append(" ".join(f"[{c}]" for _, c in chords))
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def looks_like_chordpro(text: str) -> bool:
    return bool(re.search(r"\[[A-G][^\]]*\]\S|\{\s*(title|t|start_of_chorus|soc|comment|c)\b", text))


def normalise_word(w: str) -> str:
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^\w']", "", w)
    return w.replace("'", "")


def parse_chordpro(text: str) -> list[ChartLine]:
    lines: list[ChartLine] = []
    section = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        d = _DIRECTIVE_RE.match(line)
        if d:
            name, value = d.group(1).lower(), (d.group(2) or "").strip()
            if name in ("comment", "c", "comment_italic", "ci", "comment_box", "cb", "section"):
                section = value or section
            elif name in ("start_of_chorus", "soc"):
                section = value or "Chorus"
            elif name in ("start_of_verse", "sov"):
                section = value or "Verse"
            elif name in ("start_of_bridge", "sob"):
                section = value or "Bridge"
            elif name.startswith("end_of") or name in ("eoc", "eov", "eob"):
                pass
            continue
        if _SECTION_RE.match(re.sub(r"\[([^\]]*)\]", r"\1", line)) and not any(
                is_chord(c) for c in re.findall(r"\[([^\]]+)\]", line)):
            section = line.strip().strip("[]():").strip()
            continue
        if _REPEAT_LINE.match(line) and not re.search(r"\[[^\]]+\]", line):
            continue

        # Strip [chords], remembering the character offset where each sat.
        text_parts, chord_pos = [], []
        pos, cursor = 0, 0
        for m in re.finditer(r"\[([^\]]+)\]", line):
            seg = line[cursor:m.start()]
            text_parts.append(seg)
            pos += len(seg)
            chord_pos.append((pos, m.group(1).strip()))
            cursor = m.end()
        text_parts.append(line[cursor:])
        lyric = "".join(text_parts)

        word_spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", lyric)]
        word_spans = [ws for ws in word_spans
                      if normalise_word(ws[2]) and not _REPEAT_TOKEN.match(ws[2])]
        words = [w for _, _, w in word_spans]

        anchors: list[ChordAnchor] = []
        stacked: list[str] = []
        for p, content in chord_pos:
            for sym in _bracket_chords(content):    # [] holding no chord, e.g. [x2], is skipped
                harte = chord_to_harte(sym)
                if not words:
                    anchors.append(ChordAnchor(sym, harte, None))
                    continue
                idx = next((k for k, (s, e, _) in enumerate(word_spans) if e > p), None)
                if idx is None:
                    prev = anchors[-1] if anchors and anchors[-1].trailing else None
                    anchors.append(ChordAnchor(sym, harte, len(words) - 1, trailing=True,
                                               beat_offset=prev.beat_offset + 1 if prev else 0))
                    continue
                anchor = ChordAnchor(sym, harte, idx)
                if anchors and (anchors[-1].word_index, anchors[-1].trailing) == (idx, False):
                    stacked.append(anchors.pop().symbol)
                anchors.append(anchor)
        if not words and not anchors:
            continue
        lines.append(ChartLine(len(lines), section, words, anchors, lyric.strip(), stacked))
    return lines


def parse_chart(text: str) -> list[ChartLine]:
    """Parse a ChordPro chart, converting chords-over-lyrics text first if needed."""
    if not looks_like_chordpro(text):
        text = chords_over_lyrics_to_chordpro(text)
    return parse_chordpro(text)


def chart_to_chordpro(lines: list[ChartLine]) -> str:
    """Re-render parsed lines as ChordPro (written to chart.cho for inspection)."""
    out, section = [], None
    for ln in lines:
        if ln.section != section:
            section = ln.section
            out.append("")
            if section:
                out.append("{comment: %s}" % section)
        if ln.is_instrumental:
            out.append(" ".join(f"[{c.symbol}]" for c in ln.chords))
            continue
        words = list(ln.words)
        for c in ln.chords:
            if c.trailing:
                words[c.word_index] = words[c.word_index] + f" [{c.symbol}]"
            else:
                words[c.word_index] = f"[{c.symbol}]" + words[c.word_index]
        out.append(" ".join(words))
    return "\n".join(out).strip() + "\n"


def transpose_chart(lines: list[ChartLine], semitones: int) -> list[ChartLine]:
    if semitones % 12 == 0:
        return lines
    out = []
    for ln in lines:
        chords = []
        for c in ln.chords:
            sym = transpose_symbol(c.symbol, semitones)
            chords.append(ChordAnchor(sym, chord_to_harte(sym), c.word_index, c.trailing, c.beat_offset))
        out.append(ChartLine(ln.index, ln.section, ln.words, chords, ln.text, ln.stacked))
    return out


# ---------------------------------------------------------------------------
# Key check
# ---------------------------------------------------------------------------

def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def chart_pitch_profile(lines: list[ChartLine]) -> list[float]:
    profile = [0.0] * 12
    for ln in lines:
        for c in ln.chords:
            for pc in harte_pitch_classes(c.harte):
                profile[pc] += 1.0
    return profile


def transposition_scores(lines: list[ChartLine], chroma: list[float]) -> list[float]:
    """Correlation between the audio chroma and the chart transposed by 0..11 semitones."""
    profile = chart_pitch_profile(lines)
    return [_pearson([profile[(pc - k) % 12] for pc in range(12)], chroma) for k in range(12)]


def best_transposition(scores: list[float]) -> int:
    """Best shift in -5..+6 semitones (a shift of 11 is reported as -1)."""
    k = max(range(12), key=lambda i: scores[i])
    return k - 12 if k > 6 else k


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class ChordEvent:
    time: float
    harte: str
    symbol: str
    section: str
    line_index: int
    score: float
    snapped: bool
    flags: list[str] = field(default_factory=list)
    raw_time: float | None = None    # before beat snapping


@dataclass
class Segment:
    start: float
    end: float
    label: str


@dataclass
class ReviewItem:
    start: float
    end: float
    issue: str
    detail: str


@dataclass
class LineMatch:
    line_index: int
    score: float
    t_start: int        # transcript word range [t_start, t_end)
    t_end: int


@dataclass
class AlignmentResult:
    segments: list[Segment]
    events: list[ChordEvent]
    review: list[ReviewItem]
    matches: list[LineMatch]
    stats: dict


MATCH_THRESHOLD = 0.7
LOW_MATCH = 0.8
MIN_LINE_WORDS = 3
GAP_HOLD = 3.0
MIN_EVENT_GAP = 0.05
MAX_WORD_GAP = 4.0      # a line match may not span a silence longer than this
MIN_GAP = 1.5           # shorter unaligned stretches just extend the previous chord
DEFAULT_BEAT = 0.5      # seconds, when there is no beat grid


def fill_missing_word_times(raw_words: list[dict]) -> list[Word]:
    """WhisperX word dicts -> Words; words without times (numbers etc.) get interpolated."""
    words = [Word(w.get("word", "").strip(), w.get("start"), w.get("end")) for w in raw_words]
    words = [w for w in words if normalise_word(w.text)]
    known = [i for i, w in enumerate(words) if w.start is not None and w.end is not None]
    if not known:
        return []
    for i, w in enumerate(words):
        if w.start is not None and w.end is not None:
            continue
        k = bisect.bisect_left(known, i)
        prev_end = words[known[k - 1]].end if k > 0 else words[known[0]].start
        next_start = words[known[k]].start if k < len(known) else words[known[-1]].end
        w.start = w.end = prev_end if k == len(known) else (prev_end + next_start) / 2
    return words


def snap_to_beats(t: float, beats: list[float]) -> tuple[float, bool]:
    """Snap t to the nearest beat if it is within half a (local) beat period."""
    if len(beats) < 2:
        return t, False
    k = bisect.bisect_left(beats, t)
    cands = [j for j in (k - 1, k) if 0 <= j < len(beats)]
    j = min(cands, key=lambda c: abs(beats[c] - t))
    lo, hi = max(j - 1, 0), min(j + 1, len(beats) - 1)
    period = (beats[hi] - beats[lo]) / max(hi - lo, 1)
    if abs(beats[j] - t) <= period / 2:
        return beats[j], True
    return t, False


def _beat_period(t: float, beats: list[float]) -> float:
    """Local beat length around time t."""
    if len(beats) < 2:
        return DEFAULT_BEAT
    k = min(max(bisect.bisect_left(beats, t), 1), len(beats) - 1)
    return beats[k] - beats[k - 1]


def _find_candidates(line_tokens: list[str], transcript: list[str],
                     breaks: list[int]) -> list[tuple[float, int, int]]:
    """Windows of the transcript that fuzzy-match a chart line.

    breaks[k] = number of long silences before word k; a window may not contain one.
    """
    n = len(line_tokens)
    target = " ".join(line_tokens)
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(target)
    cands = []
    for w in range(max(n - 1, 1), n + 2):
        for s in range(0, max(len(transcript) - w + 1, 0)):
            if breaks[s + w - 1] != breaks[s]:
                continue
            sm.set_seq1(" ".join(transcript[s:s + w]))
            if sm.real_quick_ratio() < MATCH_THRESHOLD or sm.quick_ratio() < MATCH_THRESHOLD:
                continue
            r = sm.ratio()
            if r >= MATCH_THRESHOLD:
                cands.append((r, s, s + w))
    # Keep only local maxima: drop candidates overlapping a better one for the same line.
    cands.sort(key=lambda c: (-c[0], c[1]))
    kept: list[tuple[float, int, int]] = []
    for c in cands:
        if all(c[2] <= k[1] or c[1] >= k[2] for k in kept):
            kept.append(c)
    return kept


def _map_words(line_tokens: list[str], window: list[str]) -> list[int | None]:
    """Map each chart word to a transcript-window index (None = not heard)."""
    mapping: list[int | None] = [None] * len(line_tokens)
    sm = difflib.SequenceMatcher(None, line_tokens, window, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            # Misheard words: still sung at roughly this time, map proportionally.
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + min(int(k * (j2 - j1) / (i2 - i1)), j2 - j1 - 1)
    return mapping


def _word_times(mapping, window_words: list[Word]) -> tuple[list[tuple[float, float]], list[bool]]:
    n = len(mapping)
    times: list[tuple[float, float] | None] = [
        (window_words[j].start, window_words[j].end) if j is not None else None for j in mapping]
    interpolated = [t is None for t in times]
    known = [i for i, t in enumerate(times) if t is not None]
    for i in range(n):
        if times[i] is not None:
            continue
        k = bisect.bisect_left(known, i)
        if 0 < k < len(known):
            p, q = known[k - 1], known[k]
            a, b = times[p][1], times[q][0]
            t = a + (b - a) * (i - p) / (q - p)
        elif k == 0:
            t = window_words[0].start if not known else times[known[0]][0]
        else:
            t = window_words[-1].end if not known else times[known[-1]][1]
        times[i] = (t, t)
    return times, interpolated


def _suggest_chords(lines: list[ChartLine], after_line: int | None, before_line: int | None) -> str:
    """Chart chords for a gap: instrumental lines between two chart positions."""
    lo = -1 if after_line is None else after_line
    hi = len(lines) if before_line is None else before_line
    if lo >= hi:            # chart order doesn't help (repeat); fall back to all instrumental lines
        lo, hi = -1, len(lines)
        pool = [ln for ln in lines if ln.is_instrumental]
    else:
        pool = [ln for ln in lines[lo + 1:hi] if ln.is_instrumental]
    parts = []
    for ln in pool:
        tag = f"{ln.section}: " if ln.section else ""
        parts.append(tag + " ".join(c.symbol for c in ln.chords))
    return " | ".join(dict.fromkeys(parts))


def align_chart(
    lines: list[ChartLine],
    words: list[Word],
    beats: list[float] | None = None,
    duration: float | None = None,
    audio_start: float = 0.0,
    audio_end: float | None = None,
) -> AlignmentResult:
    """Align chart lines to timed words and return chord segments plus review items."""
    beats = sorted(beats or [])
    review: list[ReviewItem] = []
    transcript = [normalise_word(w.text) for w in words]
    breaks, nb = [], 0
    for k, w in enumerate(words):
        if k and w.start - words[k - 1].end > MAX_WORD_GAP:
            nb += 1
        breaks.append(nb)
    duration = duration if duration is not None else (words[-1].end if words else 0.0)
    audio_end = audio_end if audio_end is not None else duration

    # Group lines with identical lyrics; same lyrics + different chords is ambiguous.
    groups: dict[tuple, list[ChartLine]] = {}
    for ln in lines:
        if ln.words:
            groups.setdefault(tuple(normalise_word(w) for w in ln.words), []).append(ln)

    candidates: list[tuple[float, int, int, int, int]] = []   # score, n_words, start, end, line
    skipped_short, ambiguous = set(), set()
    for tokens, group in groups.items():
        rep = group[0]
        if len(tokens) < MIN_LINE_WORDS:
            if rep.chords:
                skipped_short.add(rep.index)
            continue
        if len({g.chord_key() for g in group}) > 1:
            ambiguous.add(rep.index)
        for score, s, e in _find_candidates(list(tokens), transcript, breaks):
            candidates.append((score, len(tokens), s, e, rep.index))

    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    used = [False] * len(transcript)
    matches: list[LineMatch] = []
    for score, _, s, e, li in candidates:
        if any(used[s:e]):
            continue
        for k in range(s, e):
            used[k] = True
        matches.append(LineMatch(li, score, s, e))
    matches.sort(key=lambda m: m.t_start)

    matched_lines = {m.line_index for m in matches}
    for tokens, group in groups.items():
        rep = group[0]
        if rep.index in skipped_short:
            review.append(ReviewItem(0, 0, "SHORT_LINE_SKIPPED", f"line {rep.index}: {rep.text!r}"))
        elif rep.index not in matched_lines and len(tokens) >= MIN_LINE_WORDS:
            review.append(ReviewItem(0, 0, "UNMATCHED_LINE",
                                     f"line {rep.index} [{rep.section}]: {rep.text!r}"))

    # Chord events.
    events: list[ChordEvent] = []
    line_spans: list[tuple[float, float, int]] = []   # sung time span of each match
    for m in matches:
        ln = lines[m.line_index]
        window_words = words[m.t_start:m.t_end]
        tokens = [normalise_word(w) for w in ln.words]
        mapping = _map_words(tokens, transcript[m.t_start:m.t_end])
        times, interp = _word_times(mapping, window_words)
        t0, t1 = window_words[0].start, window_words[-1].end
        line_spans.append((t0, t1, m.line_index))
        if m.score < LOW_MATCH:
            review.append(ReviewItem(t0, t1, "LOW_MATCH",
                                     f"score {m.score:.2f}, line {ln.index}: {ln.text!r}"))
        if ln.index in ambiguous:
            review.append(ReviewItem(t0, t1, "AMBIGUOUS_LINE",
                                     f"line {ln.index}: same lyric has different chords in chart"))
        if ln.stacked:
            review.append(ReviewItem(t0, t1, "STACKED_CHORDS",
                                     f"dropped {', '.join(ln.stacked)} in line {ln.index}"))
        for c in ln.chords:
            ws, we = times[c.word_index]
            t = we + c.beat_offset * _beat_period(we, beats) if c.trailing else ws
            t_snap, snapped = snap_to_beats(t, beats)
            flags = []
            if interp[c.word_index]:
                flags.append("INTERPOLATED_ANCHOR")
                review.append(ReviewItem(t_snap, t_snap, "INTERPOLATED_ANCHOR",
                                         f"{c.symbol} on unheard word {ln.words[c.word_index]!r}"))
            events.append(ChordEvent(t_snap, c.harte, c.symbol, ln.section, ln.index,
                                     m.score, snapped, flags, raw_time=t))

    events.sort(key=lambda e: (e.time, e.raw_time))
    deduped: list[ChordEvent] = []
    for e in events:
        if deduped and e.time - deduped[-1].time < MIN_EVENT_GAP:
            prev = deduped[-1]
            if e.snapped and e.raw_time - prev.time >= MIN_EVENT_GAP:
                # Two chords snapped onto one beat: keep both, the later one unsnapped.
                e.time, e.snapped = e.raw_time, False
                deduped.append(e)
                continue
            deduped[-1] = e
        else:
            deduped.append(e)
    deduped.sort(key=lambda e: e.time)
    events = deduped

    # Segments.
    span_ends = sorted(line_spans)

    def sung_end_before(t: float) -> float:
        """End of the last sung line that starts at or before t."""
        k = bisect.bisect_right([s[0] for s in span_ends], t + 1e-6)
        return span_ends[k - 1][1] if k else t

    def line_at(t: float, after: bool) -> int | None:
        ks = [s for s in span_ends if (s[0] >= t if after else s[1] <= t)]
        if not ks:
            return None
        return (ks[0] if after else ks[-1])[2]

    segments: list[Segment] = []

    def add(start: float, end: float, label: str):
        if end - start > 1e-6:
            segments.append(Segment(start, end, label))

    def gap(start: float, end: float, after_line, before_line, where: str):
        if end - start <= 1e-6:
            return
        add(start, end, "X")
        sugg = _suggest_chords(lines, after_line, before_line)
        review.append(ReviewItem(start, end, "UNALIGNED_GAP",
                                 f"{where}; chart suggests: {sugg or '-'}"))

    add(0.0, audio_start, "N")
    if not events:
        gap(audio_start, audio_end, None, None, "whole song")
    else:
        if events[0].time > audio_start:
            gap(audio_start, events[0].time, None, events[0].line_index, "intro")
        for k, e in enumerate(events):
            nxt = events[k + 1].time if k + 1 < len(events) else audio_end
            hold_until = max(sung_end_before(e.time), e.time) + GAP_HOLD
            if nxt > hold_until + MIN_GAP:
                add(e.time, hold_until, e.harte)
                where = "outro" if k + 1 == len(events) else "instrumental/vamp"
                before = events[k + 1].line_index if k + 1 < len(events) else None
                gap(hold_until, nxt, e.line_index, before, where)
            else:
                add(e.time, nxt, e.harte)
    add(max(audio_end, segments[-1].end if segments else 0.0), duration, "N")

    # Merge identical neighbours.
    merged: list[Segment] = []
    for s in segments:
        if merged and merged[-1].label == s.label and abs(merged[-1].end - s.start) < 1e-6:
            merged[-1].end = s.end
        else:
            merged.append(s)

    active = max(audio_end - audio_start, 1e-9)
    labelled = sum(min(s.end, audio_end) - max(s.start, audio_start)
                   for s in merged if s.label not in ("X", "N") and s.end > audio_start
                   and s.start < audio_end)
    lyric_lines = [g[0] for t, g in groups.items() if len(t) >= MIN_LINE_WORDS]
    stats = {
        "coverage": round(labelled / active, 4),
        "lines_total": len(lyric_lines),
        "lines_found": len({m.line_index for m in matches} & {l.index for l in lyric_lines}),
        "matches": len(matches),
        "chord_events": len(events),
        "mean_match_score": round(sum(m.score for m in matches) / len(matches), 3) if matches else 0.0,
        "snapped_fraction": round(sum(e.snapped for e in events) / len(events), 3) if events else 0.0,
    }
    review.sort(key=lambda r: (r.start, r.issue))
    return AlignmentResult(merged, events, review, matches, stats)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_lab(path, segments: list[Segment]) -> None:
    with open(path, "w") as f:
        for s in segments:
            f.write(f"{s.start:.3f}\t{s.end:.3f}\t{s.label}\n")


def read_lab(path) -> list[Segment]:
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out.append(Segment(float(parts[0]), float(parts[1]), parts[2]))
    return out


def write_events_csv(path, events: list[ChordEvent]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "symbol", "harte", "section", "line", "match_score", "snapped", "flags"])
        for e in events:
            w.writerow([f"{e.time:.3f}", e.symbol, e.harte, e.section, e.line_index,
                        f"{e.score:.3f}", int(e.snapped), ";".join(e.flags)])


def write_review_csv(path, items: list[ReviewItem]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "issue", "detail"])
        for r in items:
            w.writerow([f"{r.start:.3f}", f"{r.end:.3f}", r.issue, r.detail])


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

_HARTE_CASES = {
    "G/B": "G:maj/3", "D/F#": "D:maj/3", "Am/G": "A:min/b7", "F/G": "F:maj/2",
    "Cadd9": "C:maj(9)", "C(add9)": "C:maj(9)", "Cadd2": "C:maj(9)",
    "Gadd4/B": "G:maj(11)/3", "Gadd11/B": "G:maj(11)/3", "Emadd9": "E:min(9)",
    "C2": "C:sus2", "Asus": "A:sus4", "Asus4": "A:sus4", "Asus2": "A:sus2",
    "E7sus4": "E:sus4(b7)", "E7sus": "E:sus4(b7)", "F9sus": "F:sus4(b7,9)",
    "Cmaj9": "C:maj9", "CM7": "C:maj7", "CΔ7": "C:maj7", "Cmaj7": "C:maj7",
    "Em9": "E:min9", "Am11": "A:min11", "G13": "G:13", "Em7": "E:min7",
    "A7#9": "A:7(#9)", "B7b9": "B:7(b9)", "Bbmaj7#11": "Bb:maj7(#11)",
    "F#m7b5": "F#:hdim7", "Bø": "B:hdim7", "Bø7": "B:hdim7",
    "C6/9": "C:maj6(9)", "C69": "C:maj6(9)", "Cm(maj7)": "C:minmaj7", "CmM7": "C:minmaj7",
    "C5": "C:(1,5)", "N.C.": "N", "Bdim": "B:dim", "Bdim7": "B:dim7", "B°7": "B:dim7",
    "Caug": "C:aug", "C+": "C:aug", "C6": "C:maj6", "Cm6": "C:min6", "G7": "G:7",
    "Dm7/C": "D:min7/b7", "C/D": "C:maj/2", "Gmaj13": "G:maj13", "C7#11": "C:7(#11)",
    "G13b9": "G:13(*9,b9)", "Bbm": "Bb:min", "F#": "F#:maj", "C-7": "C:min7",
    "Bdim/F": "B:dim/b5", "Cmaj7/E": "C:maj7/3", "Ebmaj7": "Eb:maj7",
}


def _run_tests() -> None:
    # Harte conversion
    for sym, want in _HARTE_CASES.items():
        got = chord_to_harte(sym)
        assert got == want, f"{sym}: got {got}, want {want}"
        parse_harte(got)                              # every output must parse back
    for bad in ("Verse", "Riff", "x2", "Hello"):
        assert not is_chord(bad), bad
    print(f"  harte: {len(_HARTE_CASES)} chord symbols OK")

    assert transpose_symbol("D/F#", -2) == "C/E"
    assert transpose_symbol("Bbmaj7", 2) == "Cmaj7"
    assert transpose_harte("G:maj/3", 2) == "A:maj/3"
    assert harte_pitch_classes("G:maj/3") == {7, 11, 2}
    assert harte_pitch_classes("C:maj(9)") == {0, 4, 7, 2}
    print("  transpose / pitch classes OK")

    # Chords-over-lyrics conversion
    col = "Verse 1\nG          D/F#\nMorning light breaking\n\nChorus\nC   G\nI will sing\nEm  C  D\n"
    cho = chords_over_lyrics_to_chordpro(col)
    assert "[G]Morning" in cho and "[D/F#]" in cho, cho
    lines = parse_chart(col)
    assert lines[0].section == "Verse 1" and lines[0].chords[0].harte == "G:maj"
    assert lines[-1].is_instrumental and [c.symbol for c in lines[-1].chords] == ["Em", "C", "D"]
    print("  chords-over-lyrics parsing OK")

    # Synthetic song: intro, verse, chorus x2, instrumental break, misheard word.
    chart = """{title: Test Song}
{comment: Intro}
[G] [D] [Em] [C]
{comment: Verse}
[G]Morning light is [D/F#]breaking through
[Em7]Every shadow [Cadd9]fades away
{comment: Chorus}
[C]You are good you [G/B]are faithful
[Am7]All my life you [Dsus]have been [D]faithful
{comment: Instrumental}
[Em] [C] [G] [D]
"""
    lines = parse_chart(chart)
    assert lines[4].stacked == [] and lines[4].chords[-1].symbol == "D"
    sung = ("morning light is breaking through every shadow fades away "
            "you are good you are faithful all my life you have been faithful "
            "you are good you are fateful all my life you have been faithful").split()
    words, t = [], 8.0
    for i, w in enumerate(sung):
        if i == 9:          # between verse and chorus
            t += 1.0
        if i == 22:         # instrumental break before the second chorus
            t += 10.0
        words.append(Word(w, t, t + 0.4))
        t += 0.5
    beats = [k * 0.5 for k in range(int(60 / 0.5))]
    res = align_chart(lines, words, beats, duration=60.0, audio_start=1.0, audio_end=55.0)
    labels = [s.label for s in res.segments]
    assert res.stats["lines_found"] == 4, res.stats
    assert res.matches and sum(1 for m in res.matches if m.line_index == 3) == 2, res.matches
    assert labels[0] == "N" and labels[1] == "X" and labels[2] == "G:maj", labels
    assert "D:maj/3" in labels and "G:maj/3" in labels and "D:sus4" in labels
    issues = [r.issue for r in res.review]
    assert issues.count("UNALIGNED_GAP") >= 3, issues     # intro, break, outro
    assert labels[-1] == "N"
    for a, b in zip(res.segments, res.segments[1:]):
        assert abs(a.end - b.start) < 1e-6 and a.label != b.label
    assert all(e.snapped for e in res.events)
    print(f"  alignment OK: coverage {res.stats['coverage']:.2f}, "
          f"{len(res.segments)} segments, issues {sorted(set(issues))}")

    # Missing words + stacked chords
    lines2 = parse_chart("[G]one two [C][D]three four\n")
    assert lines2[0].stacked == ["C"]
    words2 = [Word("one", 0.0, 0.3), Word("two", 0.5, 0.8), Word("four", 1.5, 1.8)]
    res2 = align_chart(lines2, words2, [], duration=3.0)
    assert any(e.flags == ["INTERPOLATED_ANCHOR"] for e in res2.events), res2.events
    assert any(r.issue == "STACKED_CHORDS" for r in res2.review)
    print("  interpolation / stacked chords OK")

    # Two chords one word apart must not collapse onto the same (sparse) beat.
    lines3 = parse_chart("[G]one [D]two three four\n")
    words3 = [Word(w, t, t + 0.3) for w, t in zip("one two three four".split(), (2.0, 3.0, 4.0, 5.0))]
    res3 = align_chart(lines3, words3, [0.0, 2.0, 4.0, 6.0], duration=8.0)
    assert [e.harte for e in res3.events] == ["G:maj", "D:maj"], res3.events
    print("  beat-collision OK")

    # Patterns from real WorshipTogether charts: bar notation in brackets, several chords
    # after the last word, "REPEAT CHORUS" lines.
    chart4 = ("Intro\n[| Ab / / Db | Ab / / / |]\n\nVerse 1\n"
              "With every breath that I am [Ab]able [Eb/G] [Fm]\nREPEAT CHORUS\n")
    lines4 = parse_chart(chart4)
    assert [c.symbol for c in lines4[0].chords] == ["Ab", "Db", "Ab"], lines4[0]
    assert len(lines4) == 2 and not lines4[1].stacked, lines4
    assert [(c.symbol, c.trailing, c.beat_offset) for c in lines4[1].chords] == [
        ("Ab", False, 0), ("Eb/G", True, 0), ("Fm", True, 1)], lines4[1].chords
    words4 = [Word(w, t, t + 0.3) for w, t in
              zip("with every breath that i am able".split(), (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0))]
    res4 = align_chart(lines4, words4, [k * 0.5 for k in range(30)], duration=15.0)
    ev = [(e.symbol, e.time) for e in res4.events]
    assert ev == [("Ab", 4.0), ("Eb/G", 4.5), ("Fm", 5.0)], ev
    print("  bar notation / trailing chords / repeat lines OK")

    # A gap shorter than MIN_GAP after the hold is absorbed into the chord, not marked X.
    lines5 = parse_chart("[G]one two three\n[C]four five six\n")
    words5 = [Word(w, t, t + 0.3) for w, t in
              zip("one two three four five six".split(), (0.0, 0.5, 1.0, 5.0, 5.5, 6.0))]
    res5 = align_chart(lines5, words5, [], duration=8.0)
    assert [s.label for s in res5.segments] == ["G:maj", "C:maj"], res5.segments
    print("  short-gap absorption OK")

    # Key check: a G-major chart against a C-major-ish chroma -> shift of +5 (or -7).
    gprof = chart_pitch_profile(parse_chart("[G]a b c [C]d [D]e f [Em]g\n"))
    chroma = [gprof[(pc + 7) % 12] for pc in range(12)]     # same chart, a 5th lower (C)
    assert best_transposition(transposition_scores(parse_chart("[G]a b c [C]d [D]e f [Em]g\n"),
                                                   chroma)) == 5
    print("  key check OK")
    print("all tests passed")


if __name__ == "__main__":
    _run_tests()
