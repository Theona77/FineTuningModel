# Labelling pipeline

Turns audio plus an untimed chord chart into timed Harte chord labels (`chords.lab`) for training.
It runs once, offline. The app never runs this code.

- `chord_utils.py`: chart parsing, chord symbol → Harte conversion, lyric-to-chord alignment.
  Standard library only. Run `python chord_utils.py` to run its self-tests.
- `pipeline.py`: runs ffmpeg, Demucs, librosa, Beat This! and WhisperX, then the alignment.

## Setup

```bash
pip install torch torchaudio          # build matching your CUDA, from pytorch.org
pip install -r ../requirements.txt
# ffmpeg must be on PATH
```

If WhisperX and Demucs conflict over torch versions, install each in its own environment and
point the pipeline at them:

```bash
python pipeline.py ... --demucs-python ~/envs/demucs/bin/python --whisperx-python ~/envs/wx/bin/python
```

Beat This! runs in-process. If it isn't installed, the pipeline falls back to librosa's beat
tracker, which has no downbeats and is less accurate. `report.json` records which one was used.

## Input

```
data/songs/goodness_of_god.mp3
data/charts/goodness_of_god.cho     # ChordPro: [G]Morning light [D/F#]breaking
data/charts/way_maker.txt           # or chords-over-lyrics text (auto-converted)
data/overrides.json                 # optional, see overrides.example.json
```

The song id is the audio file name without its extension. The chart must have the same name.

## Run

```bash
python labeling/pipeline.py --songs data/songs --charts data/charts --out out --only goodness_of_god
python labeling/pipeline.py --songs data/songs --charts data/charts --out out
python labeling/pipeline.py ... --only way_maker --force      # redo every step for one song
python labeling/pipeline.py ... --auto-transpose              # apply the detected key shift
```

Each step skips itself if its output already exists. The alignment step always reruns because it
is cheap, so after editing a chart you can rerun without `--force`.

## Outputs (`out/<song_id>/`)

| File | Contents |
|---|---|
| `chords.lab` | `start end label` in Harte syntax: the training label |
| `review.csv` | Everything that needs a human look, with timestamps |
| `chord_events.csv` | Every chord anchor: chart symbol, Harte label, section, match score |
| `beats.txt` | Beats for Sonic Visualiser (beat number in bar; `1` = downbeat) |
| `report.json` | Coverage, lines found, tuning, transposition scores, tempo |
| `chart.cho` | The chart as parsed (after any transposition) |
| `audio.wav`, `stems/`, `tuning.json`, `beats.json`, `words.json` | Cached intermediate results |

`out/summary.csv` has one row per song, sorted with the worst coverage first.

Review issue codes (`UNALIGNED_GAP`, `UNMATCHED_LINE`, `LOW_MATCH`, `INTERPOLATED_ANCHOR`,
`STACKED_CHORDS`, `AMBIGUOUS_LINE`, `SHORT_LINE_SKIPPED`, `KEY_MISMATCH`, `TUNING`) are
explained in PROJECT_NOTES.md section 7.

## Testing status

- `chord_utils` self-tests: 56 chord symbols (all accepted by `mir_eval`'s Harte parser), both
  chart formats, alignment on a synthetic song (repeated chorus, misheard word, intro, break,
  outro), interpolation, stacked chords, beat collisions and the key check.
- The pipeline was run end to end with stand-ins for Demucs and WhisperX and with the librosa beat
  fallback, and with the real models (htdemucs, Beat This! `final0`, WhisperX `large-v3`) on a
  60 s clip: about 40 s on an Apple M5 (CPU). Alignment on a real worship song with a real chart
  hasn't been checked yet.
- A long `torchcodec` / `libavutil` traceback from pyannote (WhisperX's voice detection) is
  harmless: it falls back to another audio loader.
