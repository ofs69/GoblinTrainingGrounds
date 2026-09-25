# Export and parity

```
python goblintrain.py export <project> --run mine --pack mine
```

`export` turns a checkpoint into a **pack** inside a GoblinScript
bundle and then proves the export against the Python pipeline. The
bundle is `<project>/bundle` unless `--bundle` says otherwise; the
first export writes the whole bundle, a later one adds its pack after
checking the perception matches.

## The bundle

A bundle (format 5) holds one frozen perception and any number of
packs:

```
bundle/
  manifest.json        everything the Rust side reads
  vjepa_pca.onnx       encoder + PCA projection, traced fp16
  transnet.onnx        shot boundaries
  packs/<name>/
    head.onnx          frontend + trunk + every deploy head, per chunk
    mask.onnx          the attention gate for one row (the viewport)
    env_step.onnx      one step of the envelope's AR decode
```

The pack's manifest block records the decode contract: chunk and
context lengths restated on the bundle's row grid, the styling
constants (stillness gate, dwell lock thresholds, reversal snap,
amplitude bound, sub-frame mode, …) and the envelope's seed and step
count. These values are read from `goblintrain/jepa_infer.py` and
`goblintrain/common.py` at export time, never restated by hand: the
bundle and the Python decoder are one decode, and a literal would go
stale silently the first time a constant moved.

The manifest's `checkpoint` field points at the source checkpoint,
repo-relative when it lives inside this tree, and is what `parity`
uses as its Python reference by default.

## Verification, in layers

1. **Trace-time verify** (part of `export`): every traced graph is run
   under ONNX Runtime against the torch modules it was traced from, on
   fixtures shaped like real data, and the export refuses on a
   mismatch. Graphs are traced into a staging directory and promoted
   file by file only once everything passed, so a failed export leaves
   the standing bundle untouched.
2. **Parity** (runs after export unless `--no-parity`): a prepared clip
   of your project goes through the bundle's graphs and through the
   Python pipeline; latents must correlate past the gate and every
   published track (phase, level, rails, envelope — autoregressive
   decode included — and the reversal heads) must agree row for row
   within tolerance. The level and the rails are sharpened
   expectations, so on a row where two bins nearly tie the two
   backends' fp32 accumulation order can move that one row by a tenth
   of a position; a track whose worst row is past the bar still passes
   when its mean difference stays under a tenth of the bar and its
   correlation holds, and the report prints the worst row, the mean and
   the count over the bar so those rows read as what they are. A wrong
   graph moves every row and fails both. `--clip <id>` picks the clip.
3. **`--rust`**: the same rows are handed to the built GoblinScript
   binary, closing the loop with the decoder people actually run.
4. **`goblintrain.py check`**: the standing decode-invariant check —
   fixtures pinning the constants both decoders copy, so a value moved
   on one side only cannot pass unnoticed.

## Using a pack

GoblinScript loads the bundle directory as-is, and `--model` picks a
pack by the name `--pack` gave it:

```
goblinscript --bundle projects/mine/bundle --model mine video.mp4
```

A directory passed with `--bundle` wins over the bundle a release
binary carries, so a released `goblinscript.exe` drafts with your pack
the same way a build from source does. `--models mine,v0.6.0` writes
two packs side by side, and the review page switches a script's model
or lays another pack's line under it. The latent cache keys on the
perception alone, so switching packs re-runs only the heads, not the
encode.

For a standalone binary that carries your pack without the flag, copy
the bundle to the GoblinScript checkout root and build it with
`--features embed`; its README covers that build and the release zip.
