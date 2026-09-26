# Projects and data

A **project** is one directory that holds one dataset. The repository
holds code and weights. A project holds data. Do not commit any file in a
project.

```
<project>/
  config.json         the frozen perception constants and your choices
  manifest.jsonl      one JSON line per clip: the record of what exists
  private/map.tsv     clip ID -> original path, never printed
  videos/             transcoded clips, named by ID
  scripts/            sanitized funscripts, named by ID
  meta/               per-clip probe records
  boundaries/         TransNetV2 shot boundaries per clip
  latents/            int8 latent caches per clip
  lag/                script-to-video lag sidecars (prepare --lag-fit)
  h0/                 frontend-output stores, keyed by trunk (derived)
  masks/              optional banner-mask rect sidecars, used when present
  rosters/            train.json, holdout.json (written by prepare)
  runs/               one directory per training run
  drafts/             eval and draft output
  .tx/                transactions in progress (empty when idle)
  .trash/             removed clips until purge
```

## Identity

A six-digit ID identifies a clip. `import` assigns the ID. The original
filename and path are stored only in `private/map.tsv`. No log line, cache
or manifest field contains them. Tools print IDs.

`manifest.jsonl` is the source of truth for the content of a project. Each
line holds: id, duration, action count, content signature of the script,
status. `status` checks the manifest against the files it names. It
reports a clip with files but no record, or a record with no files. It
never adopts them automatically.

## Import

`import` takes one video/script pair or a folder of pairs
(`<stem>.funscript` next to `<stem>.<video>`). For each pair, in one
transaction, it:

1. Probes the video with ffprobe. Refuses unreadable media.
2. Transcodes the video to the training grid (480p, 30 fps).
3. Sanitizes the funscript against the real video duration:
   - clamps positions to [0, 100]
   - drops negative and out-of-range timestamps
   - keeps only strictly increasing times
   - lowers strokes that are too fast. A device moves at most 600
     position units per second, so a full 0-to-100 stroke needs at least
     167 ms. For a faster stroke, the time of each action stays the same
     and the end position moves toward the start position until the
     stroke is at 600 units per second. Thus a fast stroke keeps its
     timing and loses depth.

   The project stores the sanitized script, not the original file. The
   import listing counts dropped actions, but not lowered strokes.
4. Computes the content signature of the script. Refuses a duplicate of a
   clip already in the project.
5. Moves the files into place by rename. Writes the manifest and the
   private map last.

A refusal gives its reason. It stops a folder import unless you pass
`--continue`. Pairs already committed stay.

## Transactions

Each change to a project is staged under `.tx/<txid>/`. It is applied by
rename on the same volume:

- each file is written to a temporary name, then renamed over the old file
- the manifest is written last.

After a crash, Ctrl-C or a full disk, the project is in the old state or
the new state, never a mixture. The next command removes leftover `.tx/`
directories before it does anything else.

`remove` moves a clip and all files derived from it to `.trash/<txid>/`.
It rewrites the manifest. `remove --undo` restores the last removal.
`purge` deletes the trash permanently.

## Caches

`prepare` fills missing items one clip at a time: boundaries, latents,
and the lag fit with `--lag-fit`. Each cache is written to a temporary
name. When complete, it is stamped and renamed. A stage that finds a temporary file treats it as
absent. Thus no reader sees a partial cache. Each latent cache stamps the
perception used to extract it. A reader refuses a stamp mismatch.

The `h0/` stores are derived caches of the frozen frontend output (see
`docs/TRAINING.md`). You can delete them. The cost is one rebuild pass.

## Banner masks

Some sources put a static banner (a logo, a URL, a border) on every frame
of a clip. A static overlay identifies the clip. Measurements show that
model attention settles on it. From a banner, the model learns which clip
it sees, not what the motion does. This is memorization. It does not
transfer to a clip that the model has not seen.

`masks/<id>.json` marks these regions. The tools set them to zero in the
model **input** before the frontend sees them. Each file lists rects in
normalized video coordinates (`x1`, `y1`, `x2`, `y2`, range 0–1). Each
rect has a `status`. The tools use rects with these values:

- `accepted`: a human accepted the rect.
- `auto`: a trusted border detection that no human rejected.

A grid cell is masked when a rect covers at least one fifth of the cell.

The mask is an input transform, not a loss term. Thus it must stay with
the checkpoint. A trunk trained with masks is evaluated, drafted and
exported with the same masks. The checkpoint stamp records the masks
directory. The tools pass that directory on and do not use a default. A
project without `masks/` trains and drafts the same in all other
respects.

This release does not include the curation tools that write these
sidecars. If you make sidecars by other means, the format above is the
contract.

## Lag fit and admission

By default, `prepare` takes each script as synchronized with its video.
It does not fit a lag, and it **admits** every scripted clip that has
latents to the training roster.

With `--lag-fit`, `prepare` also cross-correlates the velocity that the
released model predicts against each scripted clip that has no fit yet.
The result is the global time offset of the clip. The fit is out of
sample, because the released model never saw your clip. The sidecar in
`lag/` records the fit, its confidence, a polarity check and a drift
check. Training and eval shift the script of a clip by its fitted offset.
A clip without a sidecar is not shifted.

A clip that has a lag fit is admitted only when all of these are true:

- the fit peak is at least 0.55
- polarity is not suspect
- no drift alarm extends past one reversal tolerance (66.7 ms).

`prepare` prints the reason for each clip that it does not admit. A fit
stays with its clip. A later `prepare` without `--lag-fit` keeps the fit,
and training and eval still apply it.

- `rosters/holdout.json` is drawn one time from the admitted clips (one in
  eight), at the first `prepare` of all clips. A `prepare` of some clip
  IDs does not draw it. It is never rewritten. With fewer than 8
  admitted clips at that time, it is empty and stays empty.
- `rosters/train.json` holds the admitted clips that are not in the
  holdout. Each `prepare` rewrites it.
- A roster file keeps the ID of a removed clip. A tool that reads the
  roster skips an ID that is not in the manifest. Thus `remove --undo`
  puts the clip back in its roster.

Curation is per clip and done by a human. Use `status` to see why a clip
is in or out. Use `remove` to take a clip out. No tool decides in bulk.
