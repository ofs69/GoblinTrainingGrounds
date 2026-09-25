# Evaluation

```
python goblintrain.py eval <project> --run mine
python goblintrain.py eval <project> --run v0.6.0
python goblintrain.py eval <project> --run mine --ref v0.6.0-holdout
```

`eval` drafts the clips of a roster with a model. Then it **reads** the
written funscripts and compares them to the human scripts. The read uses
the `.funscript` output, not internal tracks.

- Drafts go to `<project>/drafts/<name>/` with a `metrics.json`. The
  default name is `<model>-<roster>`.
- If a clip already has a draft, `eval` reads it and does not draft it
  again.
- `--rescore` reads again the drafts that already have a read.

`--run` takes one of these:

- a run of the project
- a shipped release (`v0.6.0`, `v0.5.1`)
- a checkpoint path.

`--roster` defaults to `holdout`: the clips that `prepare` set aside one
time and that training never saw.

## Read output

The read reports per clip and pooled. Rule: **each mean has a tail
metric.** An average that improved while an artifact got worse counts as a
worse model. Thus the read prints failure-mode rates next to the averages,
not folded into them. The metric families:

- **Phase and level**: correlation of the drafted motion with the script
  motion, and the location of the drafted positions.
- **Speed**: drafted speed against scripted speed. Also reported:
  - share of over-speed strokes (>2x, >3x)
  - slow-passage behavior, separately
  - stub rate and broken-stroke rate.
- **Reversal timing**: distance from drafted reversals to scripted
  reversals, and the share within one frame.
- **Dwells**: precision and recall of parked passages, top and bottom
  separately.
- **Amplitude**: drafted excursion against script excursion. This shows
  amplitude collapse also where correlation looks normal.

`--ref <name>` prints a second eval directory next to this one, with
paired bootstrap intervals over the shared clips. Use it to compare a
fine-tune against its source release: same clips, same read, and a
difference with an interval instead of two separate numbers.

## Ground rules

- The holdout is drawn one time and never rewritten. Its numbers stay
  comparable across runs of the same project.
- Reads use the written funscripts. Thus every decode stage (styling,
  snapping, cleaning) is part of the measurement.
- Do not accept an artifact regression in exchange for an average gain.
  The read prints both. You make the decision.
