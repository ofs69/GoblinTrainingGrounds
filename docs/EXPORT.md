# Export and parity

```
python goblintrain.py export <project> --run mine --pack mine
```

`export` converts a checkpoint into a **pack** in a GoblinScript bundle.
Then it verifies the export against the Python pipeline.

- The bundle is `<project>/bundle` unless you set `--bundle`.
- The first export writes the full bundle.
- A later export checks that the perception matches, then adds its pack.

## Bundle

A bundle (format 5) holds one frozen perception and any number of packs:

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

The manifest block of the pack records the decode contract:

- chunk and context lengths, converted to the row grid of the bundle
- the styling constants (stillness gate, dwell lock thresholds, reversal
  snap, amplitude bound, sub-frame mode, …)
- the seed and step count of the envelope.

Export reads these values from `goblintrain/jepa_infer.py` and
`goblintrain/common.py`. Nobody copies them by hand. The bundle and the
Python decoder are one decode. A hand-copied literal becomes incorrect
without warning when a constant changes.

The `checkpoint` field of the manifest points to the source checkpoint.
The path is repo-relative when the checkpoint is inside this tree.
`parity` uses it as the default Python reference.

## Verification layers

1. **Trace-time verify** (part of `export`): ONNX Runtime runs each traced
   graph against the torch module it was traced from. The fixtures have
   the shape of real data. On a mismatch, the export stops. Graphs are
   traced into a staging directory. They are promoted file by file only
   after all checks pass. Thus a failed export does not change the
   existing bundle.
2. **Parity** (runs after export unless `--no-parity`): one prepared clip
   of your project goes through the bundle graphs and through the Python
   pipeline. Requirements:
   - latents correlate above the gate
   - each published track agrees row for row within tolerance: phase,
     level, rails, envelope (autoregressive decode included) and the
     reversal heads.

   The level and the rails are sharpened expectations. On a row where two
   bins almost tie, the two backends use a different fp32 accumulation
   order. This can move that one row by a tenth of a position. Thus a
   track whose worst row exceeds the bar still passes when both are true:
   - its mean difference is less than a tenth of the bar
   - its correlation holds.

   The report prints the worst row, the mean and the count of rows over
   the bar. A wrong graph moves every row and fails both conditions.
   `--clip <id>` selects the clip.
3. **`--rust`**: sends the same rows to the built GoblinScript binary.
   This tests the decoder that users run.
4. **`goblintrain.py check`**: the standing decode-invariant check.
   Fixtures pin the constants that both decoders copy. Thus a value
   changed on one side only fails the check.

## Use a pack

GoblinScript loads the bundle directory directly. `--model` selects a pack
by the name that `--pack` gave it:

```
goblinscript --bundle projects/mine/bundle --model mine video.mp4
```

- A directory passed with `--bundle` overrides the bundle inside a release
  binary. Thus a released `goblinscript.exe` drafts with your pack, the
  same as a build from source.
- `--models mine,v0.6.0` writes the drafts of two packs side by side. The
  review page can switch the model of a script, or show the line of
  another pack below it. `--bundle` replaces the built-in bundle. Thus
  the directory must hold both packs. Export v0.6.0 into it too
  (`export <project> --run v0.6.0`).
- The latent cache key depends only on the perception. Thus a pack switch
  runs only the heads again, not the encode.

To build a standalone binary that includes your pack without the flag:

1. Copy the bundle to the root of the GoblinScript checkout.
2. Build with `--features embed`.

The GoblinScript README describes that build and the release zip.
