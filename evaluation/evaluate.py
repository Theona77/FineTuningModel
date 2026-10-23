"""Score predicted .lab files against reference labels with mir_eval.

Use it to measure the baseline (stock BTC, Chordino, ...) before fine-tuning, and every
fine-tuned checkpoint after, on the same test songs.

    python evaluation/evaluate.py --ref labels --est predictions/btc_large --out results/btc_large.csv

Files are paired by name (<song_id>.lab, or <song_id>/chords.lab as written by the pipeline).
Frames labelled X in the reference are ignored. Scores are duration-weighted (MIREX style),
reported per song and as a total over all songs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mir_eval
import numpy as np

METRICS = ["root", "majmin", "majmin_inv", "thirds", "thirds_inv", "triads", "triads_inv",
           "tetrads", "tetrads_inv", "sevenths", "sevenths_inv", "mirex", "overseg", "underseg", "seg"]


def find_labs(root: Path) -> dict[str, Path]:
    labs = {p.stem: p for p in root.glob("*.lab")}
    labs.update({p.parent.name: p for p in root.glob("*/chords.lab")})
    return labs


def load_lab(path: Path):
    intervals, labels = mir_eval.io.load_labeled_intervals(str(path))
    return intervals, labels


def score_song(ref_path: Path, est_path: Path) -> tuple[dict, float]:
    ref_int, ref_lab = load_lab(ref_path)
    est_int, est_lab = load_lab(est_path)
    # X = unlabelled: drop from the reference so those frames carry no weight.
    keep = [i for i, l in enumerate(ref_lab) if l != "X"]
    ref_int, ref_lab = ref_int[keep], [ref_lab[i] for i in keep]
    est_lab = ["N" if l == "X" else l for l in est_lab]

    est_int, est_lab = mir_eval.util.adjust_intervals(
        est_int, est_lab, ref_int.min(), ref_int.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    merged, ref_m, est_m = mir_eval.util.merge_labeled_intervals(ref_int, ref_lab, est_int, est_lab)
    durations = mir_eval.util.intervals_to_durations(merged)
    # Remove the gaps left by the dropped X segments.
    covered = np.zeros(len(merged), dtype=bool)
    for s, e in ref_int:
        covered |= (merged[:, 0] >= s - 1e-9) & (merged[:, 1] <= e + 1e-9)
    durations = durations * covered

    scores = {}
    for name in METRICS:
        if name in ("overseg", "underseg", "seg"):
            continue
        comp = getattr(mir_eval.chord, name)(ref_m, est_m)
        scores[name] = float(mir_eval.chord.weighted_accuracy(comp, durations))
    scores["overseg"] = float(mir_eval.chord.overseg(ref_int, est_int))
    scores["underseg"] = float(mir_eval.chord.underseg(ref_int, est_int))
    scores["seg"] = min(scores["overseg"], scores["underseg"])
    return scores, float(durations.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True, help="directory of reference .lab files")
    ap.add_argument("--est", type=Path, required=True, help="directory of predicted .lab files")
    ap.add_argument("--songs", type=Path, help="optional text file of song ids (e.g. the test split)")
    ap.add_argument("--out", type=Path, help="write per-song scores to this CSV")
    args = ap.parse_args(argv)

    ref, est = find_labs(args.ref), find_labs(args.est)
    ids = sorted(set(ref) & set(est))
    if args.songs:
        wanted = {l.strip() for l in args.songs.read_text().splitlines() if l.strip()}
        ids = [i for i in ids if i in wanted]
    missing = sorted(set(ref) - set(est))
    if missing:
        print(f"warning: no prediction for {len(missing)} songs: {', '.join(missing[:10])}",
              file=sys.stderr)
    if not ids:
        print("no paired songs", file=sys.stderr)
        return 1

    rows, total, weight = [], {m: 0.0 for m in METRICS}, 0.0
    for song in ids:
        scores, dur = score_song(ref[song], est[song])
        rows.append({"song_id": song, "duration": round(dur, 2), **{m: round(scores[m], 4) for m in METRICS}})
        for m in METRICS:
            total[m] += scores[m] * dur
        weight += dur
    overall = {m: total[m] / weight for m in METRICS}

    width = max(len(m) for m in METRICS)
    print(f"{len(ids)} songs, {weight / 60:.1f} min labelled")
    for m in METRICS:
        print(f"  {m:<{width}}  {overall[m]:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["song_id", "duration", *METRICS])
            w.writeheader()
            w.writerows(rows)
            w.writerow({"song_id": "TOTAL", "duration": round(weight, 2),
                        **{m: round(v, 4) for m, v in overall.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
