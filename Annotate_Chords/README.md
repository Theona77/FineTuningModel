# Annotate_Chords: corrected chord labels

This folder holds **corrected** chord labels. The drafts in `../btc_predictions 2/` were made by
stock BTC and are often wrong: they only know major and minor chords, and their timing follows
BTC's 0.09 s frames rather than the beat. Your job is to fix them by ear and save the result here.
These files are what the model is trained and tested on, so accuracy matters more than speed.

## What you need

- The song's audio. It isn't in this repo (copyright), so get it from the project owner.
- [Sonic Visualiser](https://www.sonicvisualiser.org/) (free), or any editor that can show
  audio with timed labels.
- The draft label from `../btc_predictions 2/` (it may be inside a per-song subfolder).

## Workflow

1. Pick a song, and check that no one else has already done it (look in this folder and pull first).
2. In Sonic Visualiser: **File → Open** the audio, then **File → Import Annotation Layer** and
   choose the draft `.lab` file.
3. Listen through and fix each chord: its name, where it starts, and where it ends. Add chords BTC
   missed (7ths, sus, add9, slash chords) and delete ones that aren't there.
4. **File → Export Annotation Layer** as a `.lab` file into this folder, keeping the **same file
   name as the draft**, e.g. `00020-0001 - Terbaik.lab`. Drafts that sit in a subfolder go here
   directly, without the subfolder: `0113 - Send Revival.lab`.
5. Commit and push (one commit per song or per batch is fine):
   ```bash
   git pull
   git add "Annotate_Chords/00020-0001 - Terbaik.lab"
   git commit -m "Correct chords: Terbaik"
   git push
   ```

## File format

One chord per line: `start  end  chord`, times in seconds, separated by tabs or spaces.

```
0.000	1.850	N
1.850	5.620	G:maj
5.620	7.500	D:maj/3
7.500	9.380	E:min7
9.380	13.120	C:maj(9)
```

- Lines cover the whole song in order, with no gaps or overlaps.
- The last line ends at the end of the song.

## Writing chords (Harte syntax)

| Chord | Write | Chord | Write |
|---|---|---|---|
| G | `G:maj` or `G` | Gsus4 | `G:sus4` |
| Em | `E:min` | Gsus2 | `G:sus2` |
| G7 | `G:7` | Cadd9 | `C:maj(9)` |
| Gmaj7 | `G:maj7` | Em9 | `E:min9` |
| Em7 | `E:min7` | G9 | `G:9` |
| Bdim | `B:dim` | G13 | `G:13` |
| Bm7b5 | `B:hdim7` | Gaug | `G:aug` |
| G/B | `G:maj/3` | D/F# | `D:maj/3` |
| C/E | `C:maj/3` | C/G | `C:maj/5` |
| F/G | `F:maj/2` | E7sus4 | `E:sus4(b7)` |

Slash bass is written as the **interval from the root**: `/3` = major 3rd, `/b3` = minor 3rd,
`/5` = 5th, `/b7` = flat 7th, `/2` = 2nd. Use sharps or flats as the song's key does
(`Bb` in F major, `F#` in D major).

## Rules (so everyone labels the same way)

- **Label what is actually played**, in the key of this recording, even if a chart says otherwise.
- **sus2 vs. add9:** is the 3rd present? Yes → `maj(9)` (add9). No → `sus2`. Same for sus4 vs. add11.
- **Slash chords:** write the slash form whenever the bass plays a note other than the root.
- **Chord changes on the beat:** put the boundary where the beat is, not where BTC put it.
- **Passing chords** shorter than one beat: only label them if they're clearly played.
- **`N`** = silence, or no harmony (e.g. talking, a drum-only section).
- **`X`** = harmony is there but you can't tell what it is. Use it instead of guessing:
  `X` sections are skipped in training, while a wrong guess teaches the model mistakes.

The full guideline is in section 9 of `../PROJECT_NOTES.md`.
