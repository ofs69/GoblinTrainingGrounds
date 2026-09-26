# Agent guide

This repository trains, evaluates and exports the models that GoblinScript
uses to draft funscripts. `goblintrain.py` is the only entry point.
`README.md` specifies what it does.

A **project directory** holds the dataset of one user:

- media, scripts, caches, runs and drafts
- a private map from clip IDs to the original files.

A project is the directory `projects/<name>`. The commands take the
name, not a path. Do not commit a project.

## Guardrails

- Do not print an original media filename or path. Identify a clip by its
  sanitized ID.
- Do not read, write, copy, move, print or commit `private/map.tsv` or any
  file under the `private/` directory of a project. Only `import` and
  `remove` write to it. Only the human runs them.
- Do not list the files in `videos/`, `scripts/`, `latents/`, `h0/` or any
  other cache directory of a project. To inspect a project, use
  `goblintrain.py status`, `manifest.jsonl` and the `meta/` records.
- Do not send media, scripts, latents or checkpoints off the machine. Do
  not send them to a cloud service.
- Write output only inside the project directory or the repository tree.
  Do not use the session scratchpad.
- The perception configuration is frozen. Training and deploy use the same
  configuration: the encoder, the whitened PCA basis, the grid and the row
  rate. Do not re-extract a cache. Do not refit the basis. Do not change
  the grid. `prepare` fills only missing items.
- Supervise only from the real funscript. Do not use pseudo-labels,
  cluster targets, geometry targets or targets derived from validation.
- Make every change to a project through a transaction
  (`goblintrain/tx.py`):
  1. Stage the change.
  2. Commit by rename.
  3. Recover at start.

  Do not write a manifest, a map, a roster or a cache in place.
- Report each mean with a tail metric. Examples: event rate per minute,
  p99, worst clips, share outside a band. Do not accept an artifact
  regression in exchange for an average gain.
- Only a human decides to:
  - publish a release
  - change a shipped pack
  - change a recipe
  - remove clips from a project.

## Working practice

- Read `--help` before you run a command.
- Before a user trains, compare the scripted hours of their training
  roster with the approximately 105 scripted hours of v0.6.0. Much less
  footage gives a model that is worse than v0.6.0 (`docs/TRAINING.md`,
  "Corpus size"). The recommended start for a custom dataset is 5 to 10
  hours of scripted footage. Tell the user this. If they only want
  drafts, point them to `draft` or GoblinScript.
- Evaluate every run against the release on the holdout
  (`eval --run v0.6.0`, then `eval --run <run> --ref v0.6.0-holdout`).
  Report the result with its tails. If the run does not beat the
  release, say so and recommend the release.
- Validate with one of these:
  - `tests/test_tx.py`
  - `tests/smoke.py` on a small project
  - a slice of a project, selected by ID.
- Change only what the task needs. Remove dead code that a change creates.
- Documentation and comments describe the current system, not its history.
