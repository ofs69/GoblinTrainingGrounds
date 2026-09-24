# The model

One GoblinScript model is a small temporal network that reads frozen
video features and publishes, per latent row, everything the funscript
decode needs. The heavy lifting — seeing the video — is done once by a
frozen perception shared by every model; training only ever fits the
temporal network and its heads.

## The frozen perception

- **Encoder**: V-JEPA 2.1 ViT-B over 384×384 frames, 32-frame windows,
  fetched through torch.hub. Its patch tokens are projected by a frozen,
  whitened **PCA basis** (`weights/pca_basis.npz`) to 64 dimensions per
  cell of a 24×24 grid.
- **Row grid**: one latent row per 33.333 ms (30 rows/s). Every cache,
  checkpoint and sidecar carries a `row_hz` stamp and readers trust the
  stamp, not the constant.
- **Storage**: latents are cached as int8 at a fixed scale. The whitening
  makes them ~unit variance, so int8 loses nothing that matters.
- **Boundaries**: TransNetV2 shot boundaries, cached per clip. The model
  receives a per-row `cut` flag so a scene cut is information, not
  phantom motion.

The perception is frozen in `goblintrain/project.py::PERCEPTION` and
stamped into every cache. Nothing in this repository re-extracts, refits
the basis or changes the grid; a cache that disagrees is refused, not
rebuilt.

## The trunk

`frontend` (mask net → attention pooling → 1×1 input projection) turns
each row's 24×24 grid into one feature vector. The mask net scores every
cell, attention pooling averages under those weights, and the pooled
mass and centroid ride along as extra channels; the `cut` flag joins
last. The frontend is deterministic — a pure function of the latents —
which is what makes the h0 store (`docs/TRAINING.md`) possible.

On top runs a dilated temporal convolution stack (the TCN), the model's
only stochastic stage (dropout). Its features feed every head.

## The heads

Per row, the model publishes:

- **Velocity (phase)** — a categorical distribution over signed stroke
  velocity; its expectation is the marginal velocity track, the carrier
  everything else styles.
- **Level (position)** — where the stroke sits between 0 and 100,
  decoded with a sharpening temperature.
- **Extremity rails** — the low and high band the script actually
  reaches locally.
- **Dwell** — 3-class (none / top / bottom): is the stroke parked at an
  extreme. The decode's level lock reads it.
- **Reversal events** — 3-class (none / peak / valley), re-localizing
  reversal times and driving the alternating event decode.
- **Envelope** — the stroke amplitude, decoded autoregressively over a
  short context buffer. The released models use the **flow** variant: it
  integrates a small learned field from a deterministic base draw keyed
  to the absolute row, so Python and the exported graph sample
  identically without sharing RNG state.
- **Confidence** — trained so its window mean tracks the phase
  correlation the model expects against a human script. Styling ignores
  it; review tooling renders it.

## From tracks to a funscript

`goblintrain/jepa_infer.py` composes the tracks into actions: the
envelope rescales the carrier without moving its zero crossings,
reversal events snap and time the direction changes (with sub-frame
apexes), the rails and the dwell lock shape excursions, an amplitude
bound caps each stroke against the marginal's own travel, and the
result is RDP-cleaned and written as `<stem>.funscript`. The decode
constants live at the top of that module and are the deploy contract:
the exported bundle records the same values in its manifest so the Rust
decoder and the Python one are one decode (`docs/EXPORT.md`).

## The released models

Two checkpoints ship (`goblintrain.py fetch`):

- **v0.5.1** — trained with a negative pool (a retired mechanism).
- **v0.6.0** — the same recipe without negatives; ranks equal or better
  and is the default.

Each is a full model (trunk + heads). The matching bare trunks
(`<release>-trunk.pt`) are what `train --from <release>` refits heads
on. Every checkpoint carries its recipe, `v_std`, epoch, basis id and
row rate, and the loader refuses a mismatch rather than guessing.
