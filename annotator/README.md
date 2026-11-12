# annotator: Chordify-style chord editor

A local web app for correcting the draft labels in `../btc_predictions 2/` by ear on a beat grid.
It only reads and writes `.lab` files; it never touches the model code in `training/` or
`evaluation/`.

## Run

Needs Python 3.10+, `ffmpeg`, and `deno` (used by yt-dlp for YouTube). It has its own `.venv`, so
it doesn't affect the training environment.

```bash
brew install python@3.12 ffmpeg deno   # once (Mac)
annotator/run.sh
```

Then open http://127.0.0.1:8765. The first run creates `annotator/.venv` automatically.
If YouTube downloads start failing, update yt-dlp: `annotator/.venv/bin/pip install -U yt-dlp`.

## Workflow

1. Pick a song from the list (the `.lab` files in `btc_predictions 2/`) and paste its YouTube
   link, then click **Muat**. The link is remembered in `annotator/songs.json`, so the next time
   (or for a teammate) it fills in automatically and everyone uses the same recording.
2. The draft is placed on the detected beat grid. Fix the chords by ear.
3. Every change is **saved automatically** (1.2 s after you stop editing) back into the same
   `.lab` file. The status next to the form shows `● Ada perubahan…` → `✓ Tersimpan`.
   Opening a file and not changing anything does not rewrite it.
4. Commit the `.lab` files (and `annotator/songs.json`) when you're done.

**Riwayat** (button in the header, and the start screen) lists the songs you opened or edited
on this machine, newest first, with when they were last saved. Click one to reopen it with its
YouTube link. It's stored in `annotator/history.json`, which is gitignored, so it's personal;
the shared "✓ dianotasi" marks in the song list come from `songs.json`.

Audio is downloaded to `annotator/tmp/` and deleted when the tab closes, when another song is
loaded, and when the server starts. It is never committed.

## Chord format

The screen uses the familiar spelling; the file is written in Harte syntax, as required by
`Annotate_Chords/README.md` and `training/chord_vocab.py`. Every save is checked with
`labeling/chord_utils.parse_harte`, so an invalid label is never written.

| On screen | In the file | On screen | In the file |
|---|---|---|---|
| `G` | `G:maj` | `C:add9` | `C:maj(9)` |
| `E:min` | `E:min` | `G9` / `E:min9` | `G:9` / `E:min9` |
| `G#7` | `G#:7` | `B:m7b5` | `B:hdim7` |
| `G:maj7` / `E:min7` | `G:maj7` / `E:min7` | `B:dim7` | `B:dim7` |
| `G:sus2` / `G:sus4` | `G:sus2` / `G:sus4` | `E7sus4` | `E:sus4(b7)` |
| `B:dim` / `G:aug` | `B:dim` / `G:aug` | `G6` / `A:min6` | `G:maj6` / `A:min6` |
| `D/F#` | `D:maj/3` | `G13` | `G:13` |

`N` = no chord. `X` (unclear harmony) and any other Harte label already in a file are kept as is.

## Editing

- Click a cell to select it and jump there. ⇧-click or ⇧-arrow extends the selection,
  double-click selects a whole chord segment, dragging across empty cells selects a range.
- Drag a chord name left/right to move where that chord starts.
- Picker at the bottom: Root, quality (Jenis), and Bass (slash chord). Changing a cell only
  changes the selection; the cells after it keep their chord.
- Keys: `Space` play/pause, `Enter` play from selection, `A–G` root, `/` then `A–G` bass
  (`B` `/` `A` = B/A; `#` before the bass note makes it sharp), `#` up a semitone (root and bass
  move together), `M` minor/major, `7`, `J` maj7, `4` sus4, `2` sus2, `N` no chord, `⌫` merge
  into the chord on the left, `L` loop selection, `⌘Z` / `⇧⌘Z` undo/redo, `⌘⇧C` copy the labels
  (on-screen spelling). The Chord box accepts the same spelling (`B/A`, `Bm/A`, `D/F#`).
  Changing the root keeps the bass note.
- Grid: tempo ½×/2× if the beat tracker picked the wrong speed, Manual mode (BPM + first beat),
  ½-beat resolution for chords that change between beats, and Geser bar to line up beat 1.

## Configuration

`LAB_DIR` (default `../btc_predictions 2`), `SONGS_JSON` (default `annotator/songs.json`) and
`HISTORY_JSON` (default `annotator/history.json`) can be set as environment variables, e.g. to try the editor on a scratch copy of the labels.
