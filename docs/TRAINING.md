# Training

`train` has three paths:

- new heads on the released trunk (`--from`)
- fine-tune the trunk from the released trunk (`--init-from`)
- train a new trunk (`--recipe`).

Start with the first path.

## Corpus size

On all three paths, the extrema heads start from a random init and learn
from the training roster alone. The measure of size is hours of scripted
footage: training draws its windows from the scripted rows, and unscripted
gaps do not supervise. v0.6.0 trained on 367 clips: 117 hours of video,
105 hours of it scripted. Much less footage gives a model that drafts
worse than v0.6.0:

- 21 minutes (3 clips, 11 training windows), `--from v0.6.0`: worse
  than v0.6.0 on every metric, in-sample. Reversal recall at 67 ms went
  from 0.558 to 0.316, and fast strokes per minute from 12.8 to 84.2.
  The ext loss was still falling steeply at the last epoch.

The break-even amount of footage is not measured. For a custom dataset,
start with 5 to 10 hours of scripted footage. Compare each run with the
release on the holdout before you use it:

```
python goblintrain.py eval <project> --run v0.6.0
python goblintrain.py eval <project> --run mine --ref v0.6.0-holdout
```

The holdout is drawn at the first `prepare` of all clips and never
changes. A project that had fewer than 8 clips at that time has an empty
holdout and cannot make this comparison.

## New heads on the released trunk

```
python goblintrain.py train <project> --from v0.6.0 --name mine
```

`--from <release>` refits the deploy heads on the frozen bare trunk of
that release, against your training roster. `fetch` downloads the trunk
to `weights/checkpoints/<release>-trunk.pt`.

- The trunk and its gate do not change.
- Only the heads train: rails, envelope, dwell, reversal. The envelope
  embedding starts from the trunk's own modules. The rails, the envelope
  flow, the dwell and the reversal heads start from a random init, not
  from the heads of the release.
- An epoch takes minutes, not hours.

The released `v0.6.0.pt` was made from its trunk with this path. This
command reproduces it tensor for tensor, except for CUDA non-determinism
in the flow head.

## Fine-tune the trunk

```
python goblintrain.py train <project> --init-from v0.6.0 --name mine-trunk --set trunk.lr=3e-5 --set trunk.epochs=5
```

`--init-from <release>` runs the trunk stage of the recipe from the trunk
weights of that release, not from a random init. Then it refits the heads
as usual. The trunk changes. The rule in "Corpus size" applies as for
`--from`.

Set:

- a fraction of the recipe learning rate (the released trunk trained at
  3e-4)
- a small number of epochs.

Thus the run adjusts the release and does not retrain it. The checkpoint
must have the same basis and row grid as the project. The load asserts
this.

Measurement on the roster of the release (c367, 5 epochs at 3e-5, RTX
4090):

- The trunk holds the released level from epoch 1 (val mean 0.884, the
  best value of the release).
- The rank-roster draft is identical to the release within the paired
  bootstrap, tails included.

Fine-tuning on the training data of the release changes nothing. This
shows that the path does no harm. A gain must come from new clips.

Cost on 367 clips:

1. Trunk: approximately nine minutes per epoch.
2. h0 store rebuild for the new trunk: approximately twelve minutes.
3. Heads: a few minutes.

The run resumes like other runs. Its trunk checkpoint records the release
it started from.

## New trunk

```
python goblintrain.py train <project> --recipe recipes/v0.6.0.json --name fresh
```

Trains the trunk under `runs/<name>/trunk/` first, then its heads. This is
the most expensive path. It gives a gain only on a corpus of sufficient
size.

## Recipes

A recipe is a JSON file with a `trunk` section and a `heads` section. They
hold the settings that the released models used:

```json
{
 "name": "mine",
 "trunk": {"epochs": 30, "win": 6144, "stride": 3072, "batch": 1,
           "cap_two_view": true, "keep_pareto": true},
 "heads": {"epochs": 16, "win": 6144, "env_flow": true,
           "quiet_thr": 0.45, "rev_cnt_w": 0.3, "seed": 888}
}
```

- `--set section.key=value` overrides one key from the command line.
- A key that the stage does not know stops the run with an error. To
  list the valid keys, run `python -m goblintrain.jepa_train --help`
  (trunk) or `python -m goblintrain.jepa_refit --help` (heads). Write
  `--cap-two-view` as `cap_two_view`.
- Each run records its resolved settings in its directory.

`recipes/v0.5.1.json` and `recipes/v0.6.0.json` are the released recipes.
`recipes/smoke.json` has the same shape with two epochs and short windows,
for the smoke test.

## Mechanics

- **Resume.** A run writes state after each epoch. Run the same `train`
  command again to continue from the last completed epoch. The resume
  state of the heads has a fingerprint of all settings that a rerun could
  change: ids, window, batch, epochs, lr, the two head knobs. On a
  mismatch, the run stops. It does not mix two configurations.
- **Selection.** Validation uses held-out segments inside each clip. All
  scorers use this one validation definition. Scoring uses the marginal
  decode. The kept checkpoint has the best mean validation correlation. A
  slow-band read and a level read are recorded next to it.
- **h0 store.** The frontend is frozen and pointwise in time. Thus, for a
  given trunk, its output per row does not change. `train --from`
  computes it one time per clip into `<project>/h0/<key>/`. The key
  depends on the frontend weights and the feature configuration. Later
  refits on that trunk compute only the TCN and the heads. The store is
  derived and you can delete it. Each file stamps its inputs and rebuilds
  when they change.
- **Supervision.** Only from the real funscript. If the clip has a lag
  fit (`prepare --lag-fit`), the script is shifted by the fitted lag.
  Unscripted gaps and validation segments do not supervise.
- **Determinism.** Data order, RNG draws and batch shapes are fixed. Thus
  a repeated run gives the same result. The one known exception is CUDA
  kernel non-determinism in the envelope flow head, at the 1e-3 scale.
