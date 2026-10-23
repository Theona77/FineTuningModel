# Worship & Gospel Chord Recognition: Project Notes

Notes for fine-tuning a BTC (Bi-directional Transformer for Chord recognition) model on worship and
gospel music. The goal is a model that recognises the extended and slash chords common in this genre
(G/B, Cadd9, Gadd11, E7sus4, maj9, 13ths) and outputs them with timestamps.

Contents:
1. [Song list (50 songs)](#1-song-list-50-songs)
2. [Existing chord datasets](#2-existing-chord-datasets)
3. [Where to get chord charts](#3-where-to-get-chord-charts)
4. [BTC: what it covers and what it doesn't](#4-btc-what-it-covers-and-what-it-doesnt)
5. [Changes needed for fine-tuning](#5-changes-needed-for-fine-tuning)
6. [Labelling workflow](#6-labelling-workflow)
7. [The labelling pipeline (code)](#7-the-labelling-pipeline-code)
8. [Label format: Harte syntax](#8-label-format-harte-syntax)
9. [Annotation guideline](#9-annotation-guideline)
10. [Dataset hygiene and training](#10-dataset-hygiene-and-training)
11. [Training data vs. the app](#11-training-data-vs-the-app)
12. [Suggested repo layout](#12-suggested-repo-layout)
13. [Next steps](#13-next-steps)
14. [References](#14-references)

---

## 1. Song list (50 songs)

Titles and artists only. Chord charts must come from a licensed source (see section 3).

### Contemporary worship
Expect add9, sus2/sus4 and slash chords (G/B, C/E, D/F#).

| # | Song | Artist |
|---|---|---|
| 1 | Goodness of God | Bethel Music / Jenn Johnson |
| 2 | Way Maker | Sinach |
| 3 | What a Beautiful Name | Hillsong Worship |
| 4 | Oceans (Where Feet May Fail) | Hillsong United |
| 5 | Build My Life | Housefires / Pat Barrett |
| 6 | Graves Into Gardens | Elevation Worship |
| 7 | Do It Again | Elevation Worship |
| 8 | Reckless Love | Cory Asbury |
| 9 | King of Kings | Hillsong Worship |
| 10 | Gratitude | Brandon Lake |
| 11 | Firm Foundation | Cody Carnes |
| 12 | The Blessing | Kari Jobe & Cody Carnes |
| 13 | Holy Spirit | Bryan & Katie Torwalt |
| 14 | 10,000 Reasons | Matt Redman |
| 15 | Cornerstone | Hillsong Worship |
| 16 | Living Hope | Phil Wickham |
| 17 | Raise a Hallelujah | Bethel Music |
| 18 | Who You Say I Am | Hillsong Worship |
| 19 | O Come to the Altar | Elevation Worship |
| 20 | Champion | Bethel Music |
| 21 | Jireh | Elevation Worship & Maverick City |
| 22 | Same God | Elevation Worship |
| 23 | Promises | Maverick City Music |
| 24 | Talking to Jesus | Elevation Worship & Maverick City |
| 25 | Tremble | Mosaic MSC |

### Gospel
Expect 9ths, 11ths, 13ths, many slash chords and passing chords.

| # | Song | Artist |
|---|---|---|
| 26 | Break Every Chain | Tasha Cobbs Leonard |
| 27 | Gracefully Broken | Tasha Cobbs Leonard |
| 28 | Your Spirit | Tasha Cobbs Leonard |
| 29 | Total Praise | Richard Smallwood |
| 30 | Never Would Have Made It | Marvin Sapp |
| 31 | I Need You to Survive | Hezekiah Walker |
| 32 | Every Praise | Hezekiah Walker |
| 33 | Withholding Nothing | William McDowell |
| 34 | I Give Myself Away | William McDowell |
| 35 | Alpha and Omega | Israel Houghton |
| 36 | Moving Forward | Israel Houghton |
| 37 | You Are Good | Israel Houghton |
| 38 | Intentional | Travis Greene |
| 39 | Made a Way | Travis Greene |
| 40 | You Deserve It | J.J. Hairston & Youthful Praise |
| 41 | I Smile | Kirk Franklin |
| 42 | Melodies from Heaven | Kirk Franklin |
| 43 | Love Theory | Kirk Franklin |
| 44 | Oh Happy Day | Edwin Hawkins Singers |
| 45 | Victory Belongs to Jesus | Todd Dulaney |
| 46 | Nobody Like You Lord | Maranda Curtis |
| 47 | Speak to My Heart | Donnie McClurkin |
| 48 | Stand | Donnie McClurkin |
| 49 | Yes | Shekinah Glory Ministry |
| 50 | Worth | Anthony Brown & group therAPy |

Songs 29, 41–43, 47 and 48 have the richest extended harmony.

The chords you get depend on the arrangement: the same song can be charted as plain triads or with full
gospel voicings. Use one chart source per song, and the chart for the exact recording you have.

---

## 2. Existing chord datasets

There is **no public timestamped chord dataset for worship or gospel music** that we know of. These
general datasets are useful for pretraining or for mixing in:

| Dataset | Timed? | Notes |
|---|---|---|
| McGill Billboard | Yes | ~900 pop songs. Offers HTK-style `.lab` files (onset, offset, label), the same format BTC uses |
| Isophonics | Yes | Beatles, Queen, etc. Part of BTC's original training set |
| RWC-Pop | Yes | 100 Japanese-pop-style songs; audio must be obtained separately |
| USPop2002 | Yes | Part of BTC's original training set |
| JAAH | Yes | Jazz; useful for extended chords |
| ChoCo | Mixed | Integrates 18 chord sources into one JAMS format; good for pretraining |
| Chordonomicon | No | 666k+ songs with user-generated chord progressions. Untimed, but can be filtered for Christian/gospel artists for chord-transition statistics |
| HookTheory | Partly | Pop songs with Roman-numeral analysis |

Most of these do not distribute audio; you get the annotations and must source the audio yourself.

---

## 3. Where to get chord charts

| Source | Cost | Notes |
|---|---|---|
| WorshipTogether.com | Free | Charts for many contemporary worship songs |
| PraiseCharts | Paid | Detailed charts; often a ChordPro export, which is easy to parse |
| CCLI SongSelect | Licensed | ChordPro export |
| Ultimate Guitar / e-chords | Free | Noisy, user-submitted; check carefully |
| Chordify | Free/paid | Automatic chord estimates with timing. Not ground truth; useful only as a candidate |

Charts are chords over lyrics (untimed). The pipeline in section 7 turns them into timed labels.

---

## 4. BTC: what it covers and what it doesn't

BTC (Park et al., ISMIR 2019) has two output vocabularies:

- **Small:** 25 classes = 12 major + 12 minor + N (no chord).
- **Large:** 170 classes = 12 roots × 14 qualities + N + X (unknown).
  The 14 qualities: maj, min, dim, aug, maj6, min6, 7, maj7, min7, minmaj7, dim7, hdim7 (m7♭5), sus2, sus4.

### Chords BTC does not cover

| Missing | Examples | What BTC outputs instead |
|---|---|---|
| Inversions / slash chords | G/B, C/E, D/F#, F/A | Root-position chord (G, C, D) |
| Non-chord bass / pedal | C/D, F/G, Am/G | Wrong root, or X |
| Add chords | Cadd9, Gadd11, Dadd4 | C, G, or sus2/sus4 |
| Extended chords | maj9, m9, m11, 9, 11, 13 | maj7, min7, or 7 |
| 7sus | G7sus4, F9sus (common in gospel) | sus4 or 7 |
| 6/9 | C6/9 | maj6 or maj |
| Altered dominants | 7♭9, 7#9, 7#11, 7♭13, 7alt | 7 |
| Power / no-3rd chords | C5 | Usually maj, or a guess |
| Polychords / upper structures | D/C, E/C | Mostly wrong |

Several of these sound alike: sus2 vs. add9 without the 3rd, and F/G vs. G9sus. The annotation guideline
(section 9) needs a rule for each, or the labels will be inconsistent.

### Disadvantages of BTC

1. **Small, pop/rock-biased training data:** ~1,200 songs (Isophonics, UsPop2002, Robbie Williams). Almost no gospel harmony.
2. **Severe class imbalance:** maj and min dominate the training frames, so rare qualities have poor recall even though they're in the vocabulary.
3. **No bass modelling:** inversions are discarded in training, so it can't learn slash chords at all.
4. **No beat awareness:** frame-by-frame prediction, so outputs fragment into tiny chords and boundaries don't snap to beats.
5. **Short context:** fixed ~10 s windows, so there's no song-level consistency. The same chorus can get labelled differently each time.
6. **Hard cases in worship audio:** sustained pads and pedal tones blur the chroma, live recordings drift in tuning, heavy reverb smears changes, and gospel uses fast passing chords and walk-ups.
7. **Dated architecture (2019):** newer large-vocabulary models (e.g. Jiang et al.'s Chord-CNN-LSTM) handle extensions and inversions better.

---

## 5. Changes needed for fine-tuning

- **Replace the output layer.** The 170-class softmax can't represent add9, 11ths, 13ths or slash chords.
- **Recommended: factorised heads** instead of one giant softmax over every chord:
  - root head: 12 roots + N
  - quality head: triad / seventh type (maj, min, dim, aug, sus2, sus4, maj7, min7, 7, …)
  - extension head(s): multi-label 9 / 11 / 13 and alterations
  - **bass head:** 12 pitch classes, or "same as root"

  This keeps rare combinations like "Gadd9/B" from becoming empty classes.
- **Treat `X` as "ignore"** in the loss (mask those frames), since X means "unlabelled", not a class.
- **Augment by pitch shifting** ±5 semitones and transposing the labels to match (as the original BTC training does). That multiplies 50 songs into about 550 training examples.

---

## 6. Labelling workflow

The idea: generate several independent signals, combine them, and only hand-review where they disagree.

1. **Clean audio and metadata.** One format (WAV, 44.1 or 22.05 kHz), normalised loudness. Record the exact version (studio or live, album, year).
2. **Source separation** with Demucs (htdemucs):
   - bass stem → bass notes → slash-chord labels
   - "other" stem (keys, guitar, pads) → cleaner harmony
   - removing vocals and drums alone noticeably improves any chord model
3. **Tuning estimation** (`librosa.estimate_tuning`). Live recordings often drift 10–40 cents. Check the actual key against the chart and transpose the chart if needed.
4. **Beat, downbeat and structure grid** with *Beat This!* (or madmom / `allin1`). Snap chord boundaries to beats; this alone fixes most of BTC's fragmentation. Section labels let you enforce chorus-to-chorus consistency.
5. **Lyric alignment** (the key trick for chords-over-lyrics charts):
   - WhisperX on the vocal stem → word-level timestamps
   - parse the chart → which word each chord sits above
   - match chart words to transcribed words → an approximate time for each chord
   - intros, instrumentals and vamps have no words, so they need chroma-based alignment or hand labelling
6. **Ensemble chord estimation:** BTC + Chordino (NNLS-Chroma, outputs slash chords) + madmom CRF + a large-vocabulary model. Agreement means confidence; disagreement gets flagged.
7. **Bass and extension evidence:** pitch-track the bass stem (CREPE or basic-pitch) for the lowest note per beat (G vs. G/B). Transcribe the "other" stem with basic-pitch to see whether the 9th, 11th or 13th is actually sounding.
8. **Annotation guideline**, written before labelling (section 9).
9. **Review tooling:** Sonic Visualiser with audio + `.lab` layers. Sonify labels with `mir_eval.sonify` and mix with the original; wrong chords are easier to hear than to see.
10. **Dataset hygiene** (section 10).

**Order:** Demucs → tuning and beats → WhisperX + chart alignment → ensemble and bass tracking → automatic merging → manual review of flagged segments only.

Steps 2–5 are implemented in the pipeline below. Steps 6–7 are the next things to build.

---

## 7. The labelling pipeline (code)

Two files: `pipeline.py` (runs the models) and `chord_utils.py` (pure-Python parsing and alignment,
testable without a GPU: `python chord_utils.py`).

### Setup

```bash
pip install torch torchaudio          # build matching your CUDA, from pytorch.org
pip install demucs whisperx beat-this librosa soundfile numpy
# ffmpeg must be on PATH
```

If WhisperX and Demucs conflict over torch versions, run them in separate environments. The script
skips any step whose output already exists.

### Input layout

```
songs/goodness_of_god.mp3
charts/goodness_of_god.cho      # ChordPro: [G]Morning light [D/F#]breaking
charts/way_maker.txt            # or chords-over-lyrics text, auto-converted
overrides.json                  # optional: {"way_maker": {"transpose": -2}, "x": {"lang": "id"}}
```

### Run

```bash
python pipeline.py --songs songs --charts charts --out out
python pipeline.py --songs songs --charts charts --out out --only way_maker --force
python pipeline.py ... --auto-transpose
```

Try one song first with `--only <name>` and check it in Sonic Visualiser before running all 50.

### What each step does

| Step | Tool | Output |
|---|---|---|
| 2. Stems | Demucs `htdemucs` | `stems/{vocals,drums,bass,other}.wav` |
| 3. Tuning + key check | librosa | `tuning.json`; chart-vs-audio transposition scores in `report.json` |
| 4. Beats | Beat This! (`final0`), librosa fallback | `beats.json`, `beats.txt` |
| 5. Words | WhisperX `large-v3` on the vocal stem | `words.json` |
| 6. Alignment | `chord_utils` | `chords.lab`, `chord_events.csv`, `review.csv`, `report.json` |

### How the alignment works

1. Parse the chart into lines. Each chord is anchored to the word it sits above (or marked "trailing" if it sits after the last word).
2. Slide every chart line over the transcript and score it with fuzzy matching (`difflib.SequenceMatcher`, threshold 0.7). This finds **every** sung occurrence, so repeated choruses and reordered sections are handled.
3. Accept matches greedily, best score first, with no overlapping words.
4. Inside each match, map chart words to transcript words. Words Whisper missed get interpolated times and are flagged.
5. Chord time = word start time (or word end for trailing chords). Snap to the nearest beat within half a beat.
6. Regions more than 3 s after a sung line with no next chord become `X` (unknown) and are flagged, with the chart's instrumental chords suggested. Leading and trailing silence become `N`.

### Outputs per song (`out/<song_id>/`)

| File | Contents |
|---|---|
| `chords.lab` | `start  end  label` in Harte syntax. **The training label.** |
| `review.csv` | Everything needing a human look, with timestamps |
| `chord_events.csv` | Every chord anchor: chart symbol, Harte label, section, match score |
| `beats.txt` | Beats for Sonic Visualiser (downbeats marked `1`) |
| `report.json` | Coverage, lines found, tuning, transposition scores, tempo |
| `chart.cho` | The chart as parsed |
| `stems/` | Demucs stems |

`out/summary.csv` gives one row per song, for triaging the worst songs first.

### Review issues

| Issue | Meaning / action |
|---|---|
| `UNALIGNED_GAP` | Instrumental / vamp / intro / outro. Label by hand using the suggested chords, or leave as `X` |
| `UNMATCHED_LINE` | Chart line never found in the vocals. Version mismatch, or Whisper missed it |
| `LOW_MATCH` | Line matched loosely; check the timing |
| `INTERPOLATED_ANCHOR` | The word under a chord wasn't heard; time estimated |
| `STACKED_CHORDS` | Two chords on one word; the first was dropped |
| `AMBIGUOUS_LINE` | Same lyric with different chords in the chart (e.g. modulated last chorus) |
| `SHORT_LINE_SKIPPED` | Line under 3 words; can't be aligned reliably |
| `KEY_MISMATCH` | Audio fits the chart better transposed. Add an override or use `--auto-transpose` |
| `TUNING` | Recording more than 25 cents off A440 |

### Testing status

- Tested: chord-symbol → Harte conversion (31 chord types), ChordPro and chords-over-lyrics parsing, lyric-to-chord alignment on a synthetic song (repeated chorus, misheard word, instrumental break), and the full script with stand-in models (file handling, caching, outputs).
- **Not yet tested on real models:** the Demucs, WhisperX and Beat This! calls follow their documentation but haven't been run. The first real run may need a small fix.

### Known limits

- WhisperX tends to skip repeated lines on very repetitive songs (gospel vamps sung 8×). Those become `X` gaps: lower coverage, but no wrong labels.
- Chords are anchored to the sung word, so an anticipation (chord changes on the beat before the word) can land up to half a beat late after snapping.
- Live versions often add sections the chart lacks.

---

## 8. Label format: Harte syntax

`.lab` files use Harte chord syntax, which `mir_eval` and BTC expect: `root:shorthand(extensions)/bass`,
where the bass is an interval relative to the root.

| Chart symbol | Harte label |
|---|---|
| G/B | `G:maj/3` |
| D/F# | `D:maj/3` |
| Am/G | `A:min/b7` |
| F/G | `F:maj/2` |
| Cadd9, C(add9), Cadd2 | `C:maj(9)` |
| Gadd4/B, Gadd11/B | `G:maj(11)/3` |
| Emadd9 | `E:min(9)` |
| C2 (worship charts) | `C:sus2` |
| Asus, Asus4 | `A:sus4` |
| E7sus4 | `E:sus4(b7)` |
| F9sus | `F:sus4(b7,9)` |
| Cmaj9 | `C:maj9` |
| CM7 / CΔ7 | `C:maj7` |
| Em9 / Am11 / G13 | `E:min9` / `A:min11` / `G:13` |
| A7#9 / B7b9 | `A:7(#9)` / `B:7(b9)` |
| Bbmaj7#11 | `Bb:maj7(#11)` |
| F#m7b5 / Bø | `F#:hdim7` / `B:hdim7` |
| C6/9 | `C:maj6(9)` |
| Cm(maj7) | `C:minmaj7` |
| C5 | `C:(1,5)` |
| N.C. | `N` |
| (unlabelled) | `X` |

Normalisation choices: `add2` → `(9)` and `add4` → `(11)`, so one sound always gets one label.
`C2` is treated as sus2 (no 3rd), which is the usual meaning on worship charts.

---

## 9. Annotation guideline

Decide these before labelling, and apply them consistently:

- **sus2 vs. add9:** is the 3rd present? Yes → add9. No → sus2.
- **sus4 vs. add11:** same rule with the 3rd.
- **F/G vs. G9sus:** pick one convention (e.g. always write the slash form) and stick to it.
- **Minimum chord duration:** whether passing chords shorter than one beat get their own label.
- **Pedal points:** label the upper chord with the pedal as bass (e.g. `C:maj/2` over a D pedal), or not.
- **N vs. X:** `N` = silence or no harmony; `X` = harmony present but not labelled or unidentifiable.
- **Chart vs. performance:** label what is **actually played**, not what the chart says, when they differ.

---

## 10. Dataset hygiene and training

- Validate every label with `mir_eval`'s Harte parser.
- Split train / validation / test **by song**, never by segment. Keep live and studio versions of the same song in the same split.
- Pitch-shift augmentation ±5 semitones with label transposition.
- Mask `X` frames out of the loss.
- Expect class imbalance even within worship data (maj, sus, add9 will dominate). Consider class weighting or focal loss.

---

## 11. Training data vs. the app

The pipeline is **only for building training data**. It runs once, offline, on the 50 songs.

The app does **not** run Demucs, WhisperX or the alignment; they're far too slow and heavy. At runtime the
app only runs the fine-tuned model on new audio, plus optionally a lightweight beat tracker to snap chord
boundaries to the beat.

---

## 12. Suggested repo layout

```
worship-chord-recognition/
├── README.md
├── PROJECT_NOTES.md          # this file
├── labeling/
│   ├── pipeline.py           # steps 2–6
│   ├── chord_utils.py        # parsing, Harte conversion, alignment
│   └── README.md             # pipeline usage
├── data/                     # gitignored: audio, charts, stems
│   ├── songs/
│   ├── charts/
│   └── overrides.json
├── labels/                   # reviewed .lab files (commit these)
├── training/                 # BTC fine-tuning code (to build)
├── evaluation/               # mir_eval scoring scripts (to build)
└── .gitignore                # data/, out/, stems, model checkpoints
```

Don't commit audio, chord charts or stems: they're copyrighted. Commit the code, your own `.lab`
labels and the song list.

---

## 13. Next steps

1. Collect audio and charts for the 50 songs (one chart per exact recording).
2. Run the pipeline on one song, check it in Sonic Visualiser, fix any issues in the model calls.
3. Run on all 50; triage with `summary.csv`.
4. Build steps 6–7: ensemble chord models and bass-stem pitch tracking, to fill `X` gaps and verify slash chords and extensions.
5. Write the annotation guideline (section 9) and review the flagged segments.
6. Modify BTC's output layer (factorised heads), add `X` masking and pitch-shift augmentation.
7. Fine-tune, evaluate with `mir_eval` (root, majmin, sevenths, tetrads, MIREX, and inversion metrics).

---

## 14. References

- Park et al., "A Bi-Directional Transformer for Musical Chord Recognition", ISMIR 2019 (BTC).
- Foscarin, Schlüter & Widmer, "Beat This! Accurate Beat Tracking Without DBN Postprocessing", ISMIR 2024. https://github.com/CPJKU/beat_this
- de Berardinis et al., "ChoCo: a Chord Corpus and a Data Transformation Workflow for Musical Harmony Knowledge Graphs", Scientific Data, 2023. https://doi.org/10.1038/s41597-023-02410-w
- Kantarelis et al., "CHORDONOMICON: A Dataset of 666,000 Songs and their Chord Progressions", 2024. https://arxiv.org/abs/2410.22046
- Burgoyne, Wild & Fujinaga, "An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis", ISMIR 2011 (McGill Billboard). https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)
- McVicar et al., "Using online chord databases to enhance chord recognition", Journal of New Music Research, 2011.
- Harte et al., "Symbolic Representation of Musical Chords: A Proposed Syntax for Text Annotations", ISMIR 2005.
- Demucs: https://github.com/facebookresearch/demucs
- WhisperX: https://github.com/m-bain/whisperX
- mir_eval: https://github.com/mir-evaluation/mir_eval
