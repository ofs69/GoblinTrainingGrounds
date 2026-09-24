# Agent guide

This repository trains, evaluates and exports the models GoblinScript drafts
funscripts with. `goblintrain.py` is the one entry point; `README.md` is the
contract for what it does. A **project directory** holds one user's
dataset: their media, their scripts, the caches, the runs, the drafts and a
private map from clip IDs to the original files. Projects live under
`projects/` or anywhere the user chose, and they are never committed.

## Guardrails

- Never print an original media filename or path. Refer to a clip by its
  sanitized ID.
- Never read, write, copy, move, print or commit `private/map.tsv` or any
  file under a project's `private/`. Only `import` writes it, and the
  human runs that.
- Do not list files in a project's `videos/`, `scripts/`, `latents/`, `h0/`
  or any other cache directory. Inspect a project through
  `goblintrain.py status`, its `manifest.jsonl` and its `meta/` records.
- No media, script, latent or checkpoint leaves the machine. No cloud
  service receives any of it.
- Everything a command writes stays inside the project directory or the
  repository tree. Leave the session scratchpad unused.
- The perception configuration is frozen and shared by training and
  deploy: the encoder, the whitened PCA basis, the grid and the row rate.
  Never re-extract a cache, refit the basis or change the grid. `prepare`
  fills what is missing and nothing else.
- Supervise from the real funscript only. No pseudo-labels, cluster targets,
  geometry targets or validation-derived targets.
- Every change to a project goes through a transaction (`goblintrain/tx.py`):
  stage, commit by rename, recover on start. Never write a manifest, a map,
  a roster or a cache in place.
- Report every mean with a tail companion: an event rate per minute, a p99,
  the worst clips or the share outside a band. Never trade an artifact
  regression against an average gain.
- Publishing a release, changing a shipped pack, changing a recipe and
  removing clips from a project are human decisions.

## Working practice

Read `--help` before invoking a command. Validate with `tests/test_tx.py`, with
`tests/smoke.py` on a small project, or on a slice of a project by ID. Make surgical
changes and remove dead code a change creates. Documentation and comments
describe the current system, not its history.
