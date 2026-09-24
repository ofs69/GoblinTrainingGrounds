# Evaluation

```
python goblintrain.py eval <project> --run mine
python goblintrain.py eval <project> --run v0.6.0
python goblintrain.py eval <project> --run mine --ref v0.6.0-holdout
```

`eval` drafts a roster's clips with a model and **reads** the written
funscripts — the actual `.funscript` output, not internal tracks —
against the human scripts. Drafts land in
`<project>/drafts/<name>/` (default `<model>-<roster>`) with a
`metrics.json`; a clip already drafted is only re-read, and `--rescore`
re-reads drafts that already carry a read.

`--run` takes a run of the project, a shipped release (`v0.6.0`,
`v0.5.1`) or a checkpoint path. `--roster` defaults to `holdout`: the
clips `prepare` set aside once and training never saw.

## What the read reports

Per clip and pooled, the read follows one rule: **every mean carries a
tail companion.** An average that improved while an artifact got worse
is a worse model, so rates of the failure modes are printed beside the
averages rather than folded into them. The families:

- **Phase and level** — how the drafted motion correlates with the
  script's, and where the drafted positions sit.
- **Speed** — drafted speed against scripted speed, with the share of
  over-speed strokes (>2x, >3x) and slow-passage behavior read
  separately; stub and broken-stroke rates ride beside them.
- **Reversal timing** — how far drafted reversals land from scripted
  ones, with the within-one-frame share.
- **Dwells** — precision and recall of parked passages, top and bottom
  separately.
- **Amplitude** — drafted excursion against the script's, so amplitude
  collapse is visible even where correlation looks fine.

`--ref <name>` prints another eval directory beside this one with
paired bootstrap intervals over the shared clips, which is how a
fine-tune is compared against the release it started from: same clips,
same read, difference with an interval instead of two loose numbers.

## Ground rules

- The holdout is drawn once and never rewritten; numbers on it stay
  comparable across runs of the same project.
- Reads come from written funscripts, so every decode stage — styling,
  snapping, cleaning — is inside the measurement.
- Never trade an artifact regression against an average gain. The read
  prints both; the judgment stays with you.
