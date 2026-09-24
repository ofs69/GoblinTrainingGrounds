# Training

Two paths, one command.

## Fine-tune the released model (the first-class path)

```
python goblintrain.py train <project> --from v0.6.0 --name mine
```

`--from <release>` refits the deploy heads on that release's frozen bare
trunk (`weights/checkpoints/<release>-trunk.pt`, fetched by `fetch`)
against your training roster. The trunk and its gate are inherited
untouched; only the heads (rails, envelope, dwell, reversal) train.
This is cheap — epochs are minutes, not hours — and it is how the
released `v0.6.0.pt` itself was produced from its trunk, which this
command reproduces tensor for tensor up to CUDA non-determinism in the
flow head.

## A fresh trunk

```
python goblintrain.py train <project> --recipe recipes/v0.6.0.json --name fresh
```

Trains the trunk under `runs/<name>/trunk/` first, then its heads. A
trunk run is the expensive path and earns its keep only on a corpus of
some size; start with the fine-tune.

## Recipes

A recipe is a JSON file with a `trunk` and a `heads` section holding
exactly the settings the released models used:

```json
{
 "trunk": {"epochs": 30, "win": 6144, "stride": 3072, "batch": 1,
           "cap_two_view": true, "keep_pareto": true},
 "heads": {"epochs": 16, "win": 6144, "env_flow": true,
           "quiet_thr": 0.45, "rev_cnt_w": 0.3, "seed": 888}
}
```

`--set section.key=value` overrides one key from the command line; a
key outside the pruned surface is refused rather than ignored. Every
run records the settings it resolved to in its own directory.

`recipes/v0.5.1.json` and `recipes/v0.6.0.json` are the released
recipes; `recipes/smoke.json` is the same shape at two epochs on short
windows, for the smoke test.

## Mechanics worth knowing

- **Resume.** A run writes per-epoch state; the same `train` command
  again continues from the last completed epoch. The heads' resume
  state carries a fingerprint of everything a rerun could change (ids,
  window, batch, epochs, lr, the two head knobs) and refuses a
  mismatch instead of silently blending two configurations.
- **Selection.** Validation is held-out segments inside each clip (the
  one val definition every scorer shares), scored on the marginal
  decode; the kept checkpoint is the best mean val correlation, with a
  slow-band and a level read recorded beside it.
- **The h0 store.** The frontend is frozen and pointwise in time, so
  its output per row never changes for a given trunk. `train --from`
  computes it once per clip into `<project>/h0/<key>/` (keyed by the
  frontend weights and feature configuration) and every later refit on
  that trunk pays only the TCN and heads. The store is derived and
  discardable; each file stamps the inputs it was built from and
  rebuilds itself when they change.
- **Supervision.** From the real funscript only, lag-corrected by the
  clip's fitted offset. Unscripted gaps and val segments supervise
  nothing.
- **Determinism.** Data order, RNG draws and batch shapes are pinned so
  a repeated run reproduces itself; the one known exception is CUDA
  kernel non-determinism in the envelope flow head, at the 1e-3 scale.
