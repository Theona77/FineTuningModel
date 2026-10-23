# Fine-tuning BTC

Pretrained BTC's transformer, with its 170-class softmax replaced by factorised heads
(PROJECT_NOTES.md section 5):

| Head | Classes | Notes |
|---|---|---|
| root | C … B, N | |
| triad | maj min dim aug sus2 sus4 5 | |
| seventh | none b7 7 6 bb7 | |
| extensions | b5 9 b9 #9 11 #11 13 b13 | multi-label |
| bass | 12 intervals above the root | 0 = root position |

`G:maj(9)/3` (Gadd9/B) is G + maj + none + {9} + bass 4, so rare combinations never become empty
classes. Everything except the root is relative, so pitch-shift augmentation only moves the root.
`X` frames are masked out of every head; `N` frames train only the root head.

## Files

| File | |
|---|---|
| `config.yaml` | All settings; override with `--set key=value` |
| `train.py` | Training loop: validation, checkpoints, early stopping, test predictions |
| `predict.py` | Checkpoint + audio → `.lab` |
| `chord_vocab.py` | Harte label ↔ head targets (`python training/chord_vocab.py` self-tests) |
| `features.py` | BTC's log-CQT features (verified identical to BTC's own), frame targets, pitch shift |
| `dataset.py` | Song discovery, by-song splits, random augmented windows |
| `model.py` | `FactorisedBTC` and the loss |
| `inference.py` | Overlapping-window prediction, decoding, frame metrics |
| `btc_compat.py` | Imports BTC from the submodule with numpy 2 / PyYAML 6 / torch 2.6 shims |
| `make_synthetic.py` | Synthetic songs with known chords, for testing |

## Data

Training uses every song with both audio and a reviewed label:

```
data/songs/<song_id>.m4a      +    labels/<song_id>.lab
```

Other sources (e.g. McGill Billboard, JAAH) can be mixed in by adding them to
`data.sources` in `config.yaml`.

Splits are by song, never by segment. They're created on the first run in `training/splits.json`;
commit that file so the test set stays fixed. Later songs go to train. Name different versions of
one song `<song_id>__live`, `<song_id>__acoustic`, etc. and they always land in the same split.

## Train

```bash
python training/train.py
python training/train.py --set run_name=focal --set loss.focal_gamma=2
python training/train.py --set model.init=runs/finetune/best.pt     # continue from a checkpoint
```

How it trains:
- **Start:** the backbone loads from `btc_model_large_voca.pt`. For the first 3 epochs only the new heads train, then everything trains, with the backbone at 1/10 of the heads' learning rate.
- **Augmentation:** each window gets a random pitch shift (±5 semitones) and a random gain (±6 dB).
- **Class balance:** triad, seventh and bass losses use inverse-sqrt class weights, so maj/min don't dominate. Focal loss is optional.
- **Model selection:** each epoch is validated on whole songs, and the best checkpoint is chosen by `tetrads_inv` (seventh chords including the bass note). Training stops early after 12 epochs without improvement.

Outputs go to `runs/<run_name>/`:
- `best.pt` and `last.pt` checkpoints
- `metrics.csv`, one row per epoch
- `test_metrics.json`
- `predictions/{val,test}/*.lab`

## Predict and compare with the baseline

```bash
python training/predict.py --ckpt runs/finetune/best.pt --audio data/songs --out predictions/finetune
python evaluation/run_btc.py --audio data/songs --out predictions/btc_large
python evaluation/evaluate.py --ref labels --est predictions/btc_large  --songs <test ids file>
python evaluation/evaluate.py --ref labels --est runs/finetune/predictions/test
```

## Testing status

Tested end to end on 20 synthetic 60 s songs (`make_synthetic.py`): 14 train, 3 val, 3 test.
About 4 minutes on an Apple M5 (MPS). Results on the 3 test songs:

| | root | majmin | majmin_inv | sevenths_inv | tetrads_inv |
|---|---|---|---|---|---|
| Stock BTC | 0.968 | 0.971 | 0.751 | 0.743 | 0.731 |
| Fine-tuned | 0.949 | 0.948 | 0.943 | 0.934 | 0.935 |

This shows the pipeline works and that the heads can learn slash chords and extensions. Synthetic
tones are far easier than real recordings, so expect much lower numbers on worship audio.

Try it yourself:

```bash
python training/make_synthetic.py --out cache/synthetic --songs 20
python training/train.py --set run_name=synthetic \
  --set "data.sources=[{audio: cache/synthetic/audio, labels: cache/synthetic/labels}]" \
  --set data.splits_file=cache/synthetic/splits.json
```

## Not done yet

- Offline audio pitch shifting (more realistic than shifting CQT bins).
- Beat-synchronous smoothing at prediction time. `--min-duration` is a simple stand-in.
- An option to freeze some backbone layers. With only 50 songs, watch for overfitting: val loss
  rising while train loss falls.
