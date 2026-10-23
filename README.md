# Worship & Gospel Chord Recognition

Fine-tuning BTC (Bi-directional Transformer for Chord recognition) to recognise the extended and
slash chords common in worship and gospel music (G/B, Cadd9, E7sus4, maj9, 13ths). The full plan
is in PROJECT_NOTES.md.

```
labeling/      audio + chord chart -> timed .lab labels (pipeline.py, chord_utils.py)
evaluation/    run_btc.py (stock BTC baseline), evaluate.py (mir_eval scoring)
third_party/   BTC-ISMIR19 (git submodule, includes pretrained checkpoints)
training/      BTC fine-tuning with factorised heads (train.py, predict.py)
labels/        reviewed .lab files (commit these)
data/          audio, charts, overrides (gitignored: copyrighted)
```

## Setup

```bash
git submodule update --init                          # BTC code + pretrained weights
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio                         # CUDA build from pytorch.org on a GPU machine
pip install -r requirements.txt
```

Demucs, WhisperX and Beat This! all install into the one environment without conflicts.
Their weights download on first use and are cached in `~/.cache/` (torch hub and Hugging Face):
htdemucs, Beat This! `final0`, WhisperX `large-v3` and the English wav2vec2 alignment model.

## Collecting the songs

`data/songs.csv` lists the 50 songs with the `song_id` the pipeline expects. Buy the audio
(DRM-free MP3/M4A, e.g. iTunes, Amazon or Bandcamp) and get charts from a licensed source
(WorshipTogether, PraiseCharts, SongSelect), put them in one folder, then:

```bash
python labeling/check_data.py --import ~/Downloads/worship   # copies and renames into data/
python labeling/check_data.py                                # what's still missing
```

Fill in the `recording` column (album/live version) so each chart matches its recording.

## Quick start

```bash
python labeling/chord_utils.py                       # self-tests, no dependencies
python labeling/pipeline.py --songs data/songs --charts data/charts --out out --only <song_id>
```

See [labeling/README.md](labeling/README.md) for details.

## Baseline

Before fine-tuning, score stock BTC on the reviewed labels so there's a number to beat:

```bash
python evaluation/run_btc.py --audio data/songs --out predictions/btc_large            # 170 classes
python evaluation/run_btc.py --audio data/songs --out predictions/btc_majmin --majmin  # 25 classes
python evaluation/evaluate.py --ref labels --est predictions/btc_large --out results/btc_large.csv
```

`run_btc.py` uses the submodule's code and checkpoints unmodified, with small shims so the 2019
code runs on numpy 2, PyYAML 6 and torch ≥ 2.6. It uses CUDA, then Apple MPS, then CPU.

Frames labelled `X` in the reference are ignored. Stock BTC can't output slash chords or
extensions, so expect its `*_inv` and `tetrads` scores to be the weakest; those are the metrics
fine-tuning should improve.

## Fine-tuning

```bash
python training/train.py                       # songs in data/songs with labels in labels/
python training/predict.py --ckpt runs/finetune/best.pt --audio <audio> --out <dir>
```

See [training/README.md](training/README.md), including a synthetic-data test you can run now.

Don't commit audio, chord charts or stems: they're copyrighted.
