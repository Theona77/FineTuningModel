# Worship & Gospel Chord Recognition

Fine-tuning BTC (Bi-directional Transformer for Chord recognition) to recognise the extended and
slash chords common in worship and gospel music (G/B, Cadd9, E7sus4, maj9, 13ths). The full plan
is in PROJECT_NOTES.md.

```
btc_predictions 2/   481 draft .lab files: stock BTC output for the worship song collection
labels/              corrected .lab files used for training (commit these)
training/            BTC fine-tuning with factorised heads (train.py, predict.py)
evaluation/          run_btc.py (stock BTC baseline), evaluate.py (mir_eval scoring)
labeling/            optional: audio + chord chart -> timed .lab labels (pipeline.py)
third_party/         BTC-ISMIR19 (git submodule, includes pretrained checkpoints)
runs/                training outputs (gitignored except configs)
```

Audio is not in the repo: it's copyrighted and gitignored. Keep it locally next to the labels.

## Setup

```bash
git submodule update --init                          # BTC code + pretrained weights
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio                         # CUDA build from pytorch.org on a GPU machine
pip install -r requirements.txt
```

`ffmpeg` must be on the PATH: audio is decoded with it, so MP3, M4A, WAV and FLAC all work.
Training only needs torch, librosa, mir_eval, numpy and PyYAML. Demucs, WhisperX and Beat This!
are only used by the optional labelling pipeline.

## The data

### Draft labels: `btc_predictions 2/`

One `.lab` file per song (`start  end  chord`, seconds), 297 at the top level and 184 in
per-song subfolders. Song names are mostly Indonesian and English worship songs, e.g.
`00020-0001 - Terbaik.lab`, `Send Revival/0113 - Send Revival.lab`.

These are **uncorrected stock BTC predictions** from the 25-class model: every chord is major,
minor or `N`, and every boundary sits on BTC's frame grid (10/108 s). They are a starting point
for annotation, **not** training targets. Training BTC on its own output teaches it nothing new,
and these labels contain none of the chords fine-tuning is meant to add (7ths, sus, add9, slash
chords).

### Corrected labels: `labels/`

Correct each draft while listening (e.g. in Sonic Visualiser), following the annotation guideline
in PROJECT_NOTES.md section 9, and save it to `labels/<song_id>.lab`:

- Chords in Harte syntax: `G:maj`, `D:maj/3` (D/F#), `C:maj(9)` (Cadd9), `E:min7`, `A:sus4`
- `N` = silence or no harmony, `X` = harmony present but unclear (ignored in training)
- Label what is actually played, in the key of the recording
- Snap chord changes to the beat

### Audio

Put each song's audio in one folder (e.g. `audio/`), with the **same file name stem as its
label**: `audio/<song_id>.mp3` + `labels/<song_id>.lab`. Songs without a matching label are
skipped.

## Baseline

Before fine-tuning, score stock BTC on the corrected labels so there's a number to beat:

```bash
python evaluation/run_btc.py --audio audio --out predictions/btc_large            # 170 classes
python evaluation/run_btc.py --audio audio --out predictions/btc_majmin --majmin  # 25 classes
python evaluation/evaluate.py --ref labels --est predictions/btc_large --out results/btc_large.csv
```

`run_btc.py` uses the submodule's code and checkpoints unmodified, with small shims so the 2019
code runs on numpy 2, PyYAML 6 and torch ≥ 2.6. It uses CUDA, then Apple MPS, then CPU.

Frames labelled `X` in the reference are ignored. Stock BTC can't output slash chords or
extensions, so expect its `*_inv` and `tetrads` scores to be the weakest; those are the metrics
fine-tuning should improve.

## Fine-tuning

The model keeps BTC's pretrained self-attention backbone and replaces its 170-class output with
five heads (root, triad, seventh, extensions, bass), so it can output chords like `C:maj(9)/3`.

Point the data source at your audio folder (`data.sources` in `training/config.yaml`, or on
the command line), then:

```bash
python training/train.py --set "data.sources=[{audio: audio, labels: labels}]"
python training/predict.py --ckpt runs/finetune/best.pt --audio <audio> --out <dir>
python evaluation/evaluate.py --ref labels --est runs/finetune/predictions/test
```

Songs are split by song into train / val / test (saved to `training/splits.json`; commit it so
the test set stays fixed). Songs listed under `data.test_only` in the config are never trained
on and are scored separately as `holdout`. See [training/README.md](training/README.md) for
the training details and a synthetic-data test you can run without any real data.

## Labelling from chord charts (optional)

`labeling/pipeline.py` builds timed labels from audio plus an untimed chord chart (ChordPro or
chords-over-lyrics), by separating the vocals, transcribing the words and placing each chord on
the beat where its word is sung. See [labeling/README.md](labeling/README.md).

```bash
python labeling/chord_utils.py                       # self-tests, no dependencies
python labeling/pipeline.py --songs <audio dir> --charts <chart dir> --out out --auto-transpose
```

Don't commit audio, chord charts or stems: they're copyrighted.
