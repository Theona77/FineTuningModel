"""Labelling pipeline: audio + chord chart -> timed Harte chord labels (.lab).

Steps (numbering follows PROJECT_NOTES.md section 6):
  1. audio.wav      ffmpeg: 44.1 kHz stereo WAV, loudness-normalised
  2. stems/         Demucs htdemucs: vocals, drums, bass, other
  3. tuning.json    librosa: tuning offset, chroma profile, active-audio bounds
  4. beats.json     Beat This! (final0), librosa fallback
  5. words.json     WhisperX large-v3 on the vocal stem
  6. chords.lab     chord_utils alignment (always re-run; it is cheap)

Every step skips itself if its output already exists (use --force to redo).

Usage:
  python pipeline.py --songs data/songs --charts data/charts --out out
  python pipeline.py ... --only way_maker --force
  python pipeline.py ... --auto-transpose

Demucs and WhisperX run as subprocesses. If they need conflicting torch versions, install
them in separate environments and pass --demucs-python / --whisperx-python.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chord_utils as cu  # noqa: E402

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus")
CHART_EXTS = (".cho", ".chordpro", ".chopro", ".crd", ".txt")
STEMS = ("vocals", "drums", "bass", "other")
ANALYSIS_SR = 22050
TUNING_WARN_CENTS = 25.0
KEY_MARGIN = 0.05       # flag KEY_MISMATCH if a transposition beats 0 by this much


def log(song_id: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {song_id}: {msg}", flush=True)


def default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_audio(src: Path, d: Path, force: bool) -> Path:
    out = d / "audio.wav"
    if out.exists() and not force:
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "2", str(out)],
        check=True)
    return out


def step_stems(audio: Path, d: Path, args) -> Path:
    stems_dir = d / "stems"
    if all((stems_dir / f"{s}.wav").exists() for s in STEMS) and not args.force:
        return stems_dir
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [args.demucs_python, "-m", "demucs", "-n", "htdemucs", "-d", args.device,
               "-o", tmp, str(audio)]
        subprocess.run(cmd, check=True)
        produced = Path(tmp) / "htdemucs" / audio.stem
        stems_dir.mkdir(exist_ok=True)
        for s in STEMS:
            shutil.move(str(produced / f"{s}.wav"), stems_dir / f"{s}.wav")
    return stems_dir


def _load_mono(path: Path, sr: int = ANALYSIS_SR):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def _active_bounds(y, sr: int, top_db: float = 50.0) -> tuple[float, float]:
    """First and last time the signal is within top_db of its peak (RMS)."""
    import librosa
    import numpy as np
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if not rms.size or rms.max() <= 0:
        return 0.0, len(y) / sr
    db = librosa.amplitude_to_db(rms, ref=rms.max())
    idx = np.flatnonzero(db > -top_db)
    times = librosa.frames_to_time(idx, sr=sr, hop_length=512)
    return float(times[0]), float(min(times[-1] + 512 / sr, len(y) / sr))


def step_tuning(audio: Path, stems_dir: Path, d: Path, force: bool) -> dict:
    out = d / "tuning.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    import librosa
    import numpy as np

    full = _load_mono(audio)
    harm = _load_mono(stems_dir / "other.wav") + _load_mono(stems_dir / "bass.wav")
    tuning = float(librosa.estimate_tuning(y=harm, sr=ANALYSIS_SR))     # fraction of a bin
    chroma = librosa.feature.chroma_cqt(y=harm, sr=ANALYSIS_SR, tuning=tuning)
    start, end = _active_bounds(full, ANALYSIS_SR)
    info = {
        "tuning_cents": round(tuning * 100, 1),
        "chroma_mean": [round(float(v), 4) for v in np.mean(chroma, axis=1)],
        "duration": round(len(full) / ANALYSIS_SR, 3),
        "audio_start": round(start, 3),
        "audio_end": round(end, 3),
    }
    out.write_text(json.dumps(info, indent=2))
    return info


def _beat_positions(beats: list[float], downbeats: list[float]) -> list[int]:
    """Beat number within the bar (1 = downbeat); 0 before the first downbeat."""
    down = set(round(t, 3) for t in downbeats)
    pos, k = [], 0
    for t in beats:
        if round(t, 3) in down:
            k = 1
        elif k:
            k += 1
        pos.append(k)
    return pos


def step_beats(audio: Path, d: Path, args) -> dict:
    out = d / "beats.json"
    if out.exists() and not args.force:
        return json.loads(out.read_text())
    source = "beat_this"
    try:
        from beat_this.inference import File2Beats
        f2b = File2Beats(checkpoint_path="final0", device=args.device, dbn=False)
        beats, downbeats = f2b(str(audio))
        beats, downbeats = [float(t) for t in beats], [float(t) for t in downbeats]
    except Exception as e:  # noqa: BLE001 - any failure falls back to librosa
        print(f"  Beat This! unavailable ({e!r}); falling back to librosa", flush=True)
        import librosa
        y = _load_mono(audio)
        _, frames = librosa.beat.beat_track(y=y, sr=ANALYSIS_SR)
        beats = [float(t) for t in librosa.frames_to_time(frames, sr=ANALYSIS_SR)]
        downbeats, source = [], "librosa"
    intervals = [b - a for a, b in zip(beats, beats[1:])]
    tempo = 60.0 / statistics.median(intervals) if intervals else 0.0
    info = {"source": source, "tempo": round(tempo, 2),
            "beats": [round(t, 4) for t in beats], "downbeats": [round(t, 4) for t in downbeats]}
    out.write_text(json.dumps(info))
    with open(d / "beats.txt", "w") as f:       # Sonic Visualiser: time <tab> beat-in-bar
        for t, p in zip(info["beats"], _beat_positions(info["beats"], info["downbeats"])):
            f.write(f"{t:.3f}\t{p}\n")
    return info


def run_whisperx(vocals: Path, out: Path, model_name: str, device: str, lang: str | None,
                 batch_size: int) -> None:
    """Transcribe + word-align. Runs inside whichever Python has WhisperX installed."""
    import whisperx
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=lang)
    audio = whisperx.load_audio(str(vocals))
    result = model.transcribe(audio, batch_size=batch_size, language=lang)
    language = result.get("language", lang)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device,
                             return_char_alignments=False)
    words = [{"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end"),
              "score": w.get("score")} for w in aligned.get("word_segments", [])]
    out.write_text(json.dumps({"language": language, "model": model_name, "words": words},
                              indent=1))


def step_words(stems_dir: Path, d: Path, args, lang: str | None) -> list[cu.Word]:
    out = d / "words.json"
    if not out.exists() or args.force:
        cmd = [args.whisperx_python, str(Path(__file__).resolve()), "_whisperx",
               str(stems_dir / "vocals.wav"), str(out), "--model", args.whisper_model,
               "--device", args.device, "--batch-size", str(args.batch_size)]
        if lang:
            cmd += ["--lang", lang]
        subprocess.run(cmd, check=True)
    return cu.fill_missing_word_times(json.loads(out.read_text())["words"])


def step_align(song_id: str, chart_path: Path, d: Path, words: list[cu.Word], beats: dict,
               tuning: dict, override: dict, auto_transpose: bool) -> dict:
    lines = cu.parse_chart(chart_path.read_text(encoding="utf-8", errors="replace"))
    review: list[cu.ReviewItem] = []

    scores = cu.transposition_scores(lines, tuning["chroma_mean"])
    best = cu.best_transposition(scores)
    transpose = int(override.get("transpose", 0))
    current = scores[transpose % 12]
    if best % 12 != transpose % 12 and scores[best % 12] - current > KEY_MARGIN:
        if auto_transpose and "transpose" not in override:
            transpose = best
        review.append(cu.ReviewItem(0, 0, "KEY_MISMATCH",
                                    f"audio fits chart transposed {best:+d} "
                                    f"(r={scores[best % 12]:.2f} vs {current:.2f}); "
                                    f"applied {transpose:+d}"))
    lines = cu.transpose_chart(lines, transpose)
    if abs(tuning["tuning_cents"]) > TUNING_WARN_CENTS:
        review.append(cu.ReviewItem(0, 0, "TUNING", f"{tuning['tuning_cents']:+.1f} cents off A440"))

    res = cu.align_chart(lines, words, beats["beats"], duration=tuning["duration"],
                         audio_start=tuning["audio_start"], audio_end=tuning["audio_end"])
    review = review + res.review

    (d / "chart.cho").write_text(cu.chart_to_chordpro(lines))
    cu.write_lab(d / "chords.lab", res.segments)
    cu.write_events_csv(d / "chord_events.csv", res.events)
    cu.write_review_csv(d / "review.csv", review)

    issue_counts: dict[str, int] = {}
    for r in review:
        issue_counts[r.issue] = issue_counts.get(r.issue, 0) + 1
    report = {
        "song_id": song_id,
        "chart": chart_path.name,
        **res.stats,
        "unmatched_lines": issue_counts.get("UNMATCHED_LINE", 0),
        "tuning_cents": tuning["tuning_cents"],
        "transpose_applied": transpose,
        "transposition_scores": {f"{(k if k <= 6 else k - 12):+d}": round(s, 3)
                                 for k, s in enumerate(scores)},
        "tempo": beats.get("tempo"),
        "beat_source": beats.get("source"),
        "duration": tuning["duration"],
        "words_transcribed": len(words),
        "issues": issue_counts,
    }
    (d / "report.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def find_chart(charts_dir: Path, song_id: str) -> Path | None:
    for ext in CHART_EXTS:
        p = charts_dir / f"{song_id}{ext}"
        if p.exists():
            return p
    return None


def process_song(audio_src: Path, chart: Path, out_root: Path, args, override: dict) -> dict:
    song_id = audio_src.stem
    d = out_root / song_id
    d.mkdir(parents=True, exist_ok=True)

    # Each heavy step only runs if its output is missing; upstream outputs are produced lazily.
    cache: dict = {}

    def audio():
        if "audio" not in cache:
            log(song_id, "audio")
            cache["audio"] = step_audio(audio_src, d, args.force)
        return cache["audio"]

    def stems():
        if "stems" not in cache:
            log(song_id, "stems (demucs)")
            cache["stems"] = step_stems(audio(), d, args)
        return cache["stems"]

    def need(name: str) -> bool:
        return args.force or not (d / name).exists()

    if need("tuning.json"):
        log(song_id, "tuning")
        tuning = step_tuning(audio(), stems(), d, args.force)
    else:
        tuning = json.loads((d / "tuning.json").read_text())
    if need("beats.json"):
        log(song_id, "beats")
        beats = step_beats(audio(), d, args)
    else:
        beats = json.loads((d / "beats.json").read_text())
    if need("words.json"):
        log(song_id, "words (whisperx)")
        words = step_words(stems(), d, args, override.get("lang", args.lang))
    else:
        words = cu.fill_missing_word_times(json.loads((d / "words.json").read_text())["words"])

    log(song_id, "align")
    report = step_align(song_id, chart, d, words, beats, tuning, override, args.auto_transpose)
    log(song_id, f"coverage {report['coverage']:.0%}, lines {report['lines_found']}/"
                 f"{report['lines_total']}, issues {report['issues']}")
    return report


SUMMARY_FIELDS = ["song_id", "status", "coverage", "lines_found", "lines_total", "unmatched_lines",
                  "mean_match_score", "chord_events", "tuning_cents", "transpose_applied", "tempo",
                  "beat_source", "duration", "error"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", type=Path, required=True)
    ap.add_argument("--charts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--overrides", type=Path, default=None,
                    help="JSON {song_id: {transpose: int, lang: str}} (default: <charts>/../overrides.json)")
    ap.add_argument("--only", nargs="*", help="song ids to process")
    ap.add_argument("--force", action="store_true", help="recompute every step")
    ap.add_argument("--auto-transpose", action="store_true",
                    help="apply the best-fitting chart transposition when no override is given")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--lang", default="en", help="WhisperX language (default en)")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--demucs-python", default=sys.executable)
    ap.add_argument("--whisperx-python", default=sys.executable)
    args = ap.parse_args(argv)
    args.device = args.device or default_device()

    overrides_path = args.overrides or args.charts.parent / "overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}

    songs = sorted(p for p in args.songs.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if args.only:
        songs = [p for p in songs if p.stem in set(args.only)]
    if not songs:
        print("no matching audio files", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in songs:
        chart = find_chart(args.charts, src.stem)
        row = {"song_id": src.stem}
        if chart is None:
            row.update(status="no_chart")
            log(src.stem, "no chart found, skipping")
        else:
            try:
                report = process_song(src, chart, args.out, args, overrides.get(src.stem, {}))
                row.update({k: report.get(k) for k in SUMMARY_FIELDS if k in report}, status="ok")
            except Exception as e:  # noqa: BLE001 - keep going with the other songs
                traceback.print_exc()
                row.update(status="error", error=repr(e)[:200])
        rows.append(row)

    # Merge with any previous summary so --only runs don't drop other songs.
    summary_path = args.out / "summary.csv"
    previous = {}
    if summary_path.exists():
        with open(summary_path) as f:
            previous = {r["song_id"]: r for r in csv.DictReader(f)}
    previous.update({r["song_id"]: r for r in rows})
    ordered = sorted(previous.values(),
                     key=lambda r: (r.get("status") != "ok", float(r.get("coverage") or 0)))
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    print(f"summary: {summary_path} (worst coverage first)")
    return 0 if all(r.get("status") == "ok" for r in rows) else 2


def _whisperx_main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vocals", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    run_whisperx(a.vocals, a.out, a.model, a.device, a.lang, a.batch_size)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_whisperx":
        sys.exit(_whisperx_main(sys.argv[2:]))
    sys.exit(main())
