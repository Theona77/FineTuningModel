"""Track which songs in data/songs.csv have audio and a usable chart.

    python labeling/check_data.py                          # status table
    python labeling/check_data.py --import ~/Downloads     # copy purchased files in, renamed to <song_id>

--import matches files to songs by fuzzy title (file name or "Artist - Title" tags in the name),
copies audio to data/songs/<id>.<ext> and charts (.cho/.chordpro/.txt/.pro) to data/charts/<id>.<ext>.
It never overwrites an existing file and prints what it couldn't match.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chord_utils as cu  # noqa: E402
from pipeline import AUDIO_EXTS, CHART_EXTS, find_chart  # noqa: E402

ROOT = HERE.parent
IMPORT_CHART_EXTS = CHART_EXTS + (".pro",)


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s.lower())
    s = re.sub(r"\b(official|audio|video|live|lyrics?|chords?|chart|feat\.?|ft\.?)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def match_song(name: str, songs: list[dict]) -> dict | None:
    """Best manifest row for a file name, or None if nothing is close enough."""
    n = _norm(name)
    best, best_score = None, 0.0
    for s in songs:
        title = _norm(s["title"])
        if re.search(rf"\b{re.escape(title)}\b", n):
            score = 1.0 + len(title) / 100          # prefer the longest exact title
            if _norm(s["artist"].split("/")[0]) in n:
                score += 0.5
        else:
            score = difflib.SequenceMatcher(None, title, n).ratio()
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 0.8 else None


def find_audio(songs_dir: Path, song_id: str) -> Path | None:
    return next((songs_dir / f"{song_id}{e}" for e in AUDIO_EXTS
                 if (songs_dir / f"{song_id}{e}").exists()), None)


def chart_status(path: Path) -> str:
    try:
        lines = cu.parse_chart(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        return f"parse error: {e}"
    lyric = sum(1 for l in lines if l.words)
    chords = sum(len(l.chords) for l in lines)
    if not lyric or not chords:
        return f"suspicious: {lyric} lyric lines, {chords} chords"
    return f"ok ({lyric} lines, {chords} chords)"


def _show(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def import_files(src: Path, songs: list[dict], songs_dir: Path, charts_dir: Path) -> None:
    songs_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        ext = f.suffix.lower()
        if ext in AUDIO_EXTS:
            dest_dir = songs_dir
        elif ext in IMPORT_CHART_EXTS:
            dest_dir, ext = charts_dir, (".cho" if ext == ".pro" else ext)
        else:
            continue
        song = match_song(f.stem, songs)
        if song is None:
            print(f"  no match: {f.name}")
            continue
        dest = dest_dir / f"{song['id']}{ext}"
        if dest.exists():
            print(f"  exists:   {f.name} -> {_show(dest)} (skipped)")
            continue
        shutil.copy2(f, dest)
        print(f"  imported: {f.name} -> {_show(dest)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "songs.csv")
    ap.add_argument("--songs", type=Path, default=ROOT / "data" / "songs")
    ap.add_argument("--charts", type=Path, default=ROOT / "data" / "charts")
    ap.add_argument("--import", dest="import_dir", type=Path, help="folder of purchased audio/charts")
    args = ap.parse_args(argv)

    songs = load_manifest(args.manifest)
    if args.import_dir:
        import_files(args.import_dir, songs, args.songs, args.charts)
        print()

    ready = 0
    print(f"{'#':>3}  {'song_id':<28} {'audio':<6} chart")
    for s in songs:
        audio = find_audio(args.songs, s["id"]) if args.songs.exists() else None
        chart = find_chart(args.charts, s["id"]) if args.charts.exists() else None
        status = chart_status(chart) if chart else "missing"
        ready += bool(audio and status.startswith("ok"))
        print(f"{s['num']:>3}  {s['id']:<28} {'yes' if audio else '-':<6} {status}")
    print(f"\n{ready}/{len(songs)} songs ready for the pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
