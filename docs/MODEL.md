# The model

A GoblinScript model is a small temporal network. It reads frozen video
features. For each latent row, it publishes all values that the funscript
decode needs. A frozen perception, shared by all models, processes the
video. Training fits only the temporal network and its heads.

## Frozen perception

- **Encoder**: V-JEPA 2.1 ViT-B on 384×384 frames, 32-frame windows,
  downloaded through torch.hub. A frozen, whitened **PCA basis**
  (`weights/pca_basis.npz`) projects its patch tokens to 64 dimensions for
  each cell of a 24×24 grid.
- **Row grid**: one latent row per 33.333 ms (30 rows/s). Each cache,
  checkpoint and sidecar has a `row_hz` stamp. Readers use the stamp, not
  the constant.
- **Storage**: latents are stored as int8 at a fixed scale. The whitening
  gives them approximately unit variance. Thus int8 loses no significant
  precision.
- **Boundaries**: TransNetV2 shot boundaries, cached per clip. The model
  receives a `cut` flag for each row. Thus a scene cut is an input signal,
  not false motion.

The perception is frozen in `goblintrain/project.py::PERCEPTION` and
stamped into every cache. No code in this repository re-extracts latents,
refits the basis or changes the grid. A cache that does not match is
refused, not rebuilt.

## Trunk

The `frontend` (mask net → attention pooling → 1×1 input projection)
converts the 24×24 grid of each row into one feature vector:

- the mask net scores each cell
- attention pooling averages the cells with those weights
- the pooled mass and centroid are added as extra channels
- the `cut` flag is added last.

The frontend is deterministic: a pure function of the latents. This
property enables the h0 store (`docs/TRAINING.md`).

A dilated temporal convolution stack (the TCN) runs on the frontend
output. It is the only stochastic stage of the model (dropout). Its
features go to every head.

## Heads

For each row, the model publishes:

- **Velocity (phase)**: a categorical distribution over signed stroke
  velocity. Its expectation is the marginal velocity track. This track is
  the carrier that all other outputs style.
- **Level (position)**: the position of the stroke between 0 and 100,
  decoded with a sharpening temperature.
- **Extremity rails**: the low and high band that the script reaches
  locally.
- **Dwell**: 3 classes (none / top / bottom). Tells if the stroke is
  parked at an extreme. The level lock of the decode reads it.
- **Reversal events**: 3 classes (none / peak / valley). They re-localize
  reversal times and drive the alternating event decode.
- **Envelope**: the stroke amplitude, decoded autoregressively over a
  short context buffer. The released models use the **flow** variant. It
  integrates a small learned field from a deterministic base draw keyed to
  the absolute row. Thus Python and the exported graph sample identically
  without shared RNG state.
- **Confidence**: trained so that its window mean tracks the expected
  phase correlation against a human script. Styling ignores it. Review
  tooling shows it.

## From tracks to a funscript

A funscript is a JSON file. Its `actions` list is a sequence of
**points**, each `{"at": <milliseconds>, "pos": <0..100>}`.

- The player moves the device linearly from one point to the next.
- `pos` 0 is the bottom of the stroke. `pos` 100 is the top.
- A point is written only where the straight line must bend: a reversal,
  a hold, a change of pace. Between two points, the player interpolates.

This is the output contract of the decode below. It is also the input
contract of training: the model is supervised only from these points.

`goblintrain/jepa_infer.py` composes the tracks into actions:

- The envelope rescales the carrier. It does not move the zero crossings.
- Reversal events snap and time the direction changes, with sub-frame
  apexes.
- The rails and the dwell lock shape the excursions.
- An amplitude bound limits each stroke against the travel of the
  marginal.

The result is RDP-cleaned and written as `<stem>.funscript`.

The decode constants are at the top of that module. They are the deploy
contract. The exported bundle records the same values in its manifest.
Thus the Rust decoder and the Python decoder are one decode
(`docs/EXPORT.md`).

## Released models

`goblintrain.py fetch` downloads two checkpoints:

- **v0.5.1**: trained with a negative pool (a retired mechanism).
- **v0.6.0**: the same recipe without negatives. It ranks equal or better.
  It is the default.

Each is a full model (trunk + heads). `train --from <release>` refits
heads on the matching bare trunk (`<release>-trunk.pt`). Each checkpoint
records its recipe, `v_std`, epoch, basis id and row rate. The loader
refuses a mismatch. It does not guess.
