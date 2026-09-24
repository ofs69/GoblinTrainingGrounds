# Projects and data

A **project** is one directory holding one dataset. The repository holds
code and weights; a project holds data. Nothing inside a project is ever
committed.

```
<project>/
  config.json         the frozen perception constants and your choices
  manifest.jsonl      one JSON line per clip: the record of what exists
  private/map.tsv     clip ID -> original path; yours, never printed
  videos/             transcoded clips, named by ID
  scripts/            sanitized funscripts, named by ID
  meta/               per-clip probe records
  boundaries/         TransNetV2 shot boundaries per clip
  latents/            int8 latent caches per clip
  lag/                script-to-video lag sidecars per clip
  h0/                 frontend-output stores, keyed by trunk (derived)
  masks/              optional banner-mask rect sidecars, honored when present
  rosters/            train.json, holdout.json — written by prepare
  runs/               one directory per training run
  drafts/             eval and draft output
  .tx/                transactions in flight (empty when idle)
  .trash/             removed clips until purge
```

## Identity

A clip is its six-digit ID, assigned at import. The original filename
and path go into `private/map.tsv` and nowhere else: no log line, no
cache, no manifest field repeats them. Tools print IDs.

`manifest.jsonl` is the source of truth for what a project contains:
id, duration, action count, the script's content signature, status.
Every command starts by checking the manifest against the files it
names; a clip with files but no record, or a record without files, is
reported and never silently adopted.

## Import

`import` takes one video/script pair or a folder of pairs
(`<stem>.funscript` beside `<stem>.<video>`). Per pair, inside a
transaction:

1. probe the video (ffprobe) and refuse unreadable media,
2. transcode to the training grid (480p, 30 fps),
3. sanitize the funscript against the video's real duration — clamp
   positions to [0, 100], drop negative or out-of-range timestamps,
   keep strictly increasing times,
4. compute the script's content signature and refuse a duplicate of a
   clip already in the project,
5. move the files into place by rename, then write the manifest and the
   private map last.

A refusal names its reason and stops a folder import unless
`--continue`; pairs already committed stay.

## Transactions

Every change to a project stages under `.tx/<txid>/` and lands by
rename on the same volume, manifest last, each file written to a
temporary name and renamed over the old one. A crash, Ctrl-C or full
disk leaves either the old state or the new one, never a mixture; the
next command removes leftover `.tx/` directories before doing anything.

`remove` moves a clip and everything derived from it to
`.trash/<txid>/` and rewrites the manifest; `remove --undo` restores
the last removal; `purge` empties the trash for good.

## Caches

`prepare` fills what is missing, clip by clip: boundaries, latents, the
lag fit. Every cache is written to a temporary name and renamed when
complete and stamped; a stage that finds a temporary file treats it as
absent, so nothing ever reads a partial cache. Each latent cache stamps
the perception it was extracted under, and a reader refuses a stamp
mismatch.

The `h0/` stores are derived, discardable caches of the frozen
frontend's output (see `docs/TRAINING.md`); deleting them costs one
rebuild pass.

## Banner masks

Some sources stamp a static banner — a logo, a URL, a border — onto
every frame of a clip. A static overlay is a clip-identity fingerprint:
the model's attention measurably settles on it, and what it learns from
a banner is *which clip this is*, not what the motion does. That is
memorization wearing the costume of skill, and it does not survive
contact with a clip the model has never seen.

`masks/<id>.json` marks such regions so they can be zeroed out of the
model's **input** before the frontend sees them. Each file lists rects
in normalized video coordinates (`x1`, `y1`, `x2`, `y2` in 0–1) with a
`status`; rects marked `accepted` (a human's verdict) or `auto` (a
trusted border detection nobody rejected) are honored, and a grid cell
is masked when a rect covers at least a fifth of it.

Because it is an input transform, not a loss trick, it must follow the
checkpoint: a trunk trained with masks is evaluated, drafted and
exported with them, so the checkpoint stamp records the masks
directory and the tools pass it along rather than defaulting. A project
with no `masks/` trains and drafts identically in every other respect.
The curation tools that write these sidecars do not ship in this
release; the format above is the contract if you produce them by other
means.

## The lag fit and admission

`prepare` cross-correlates the released model's predicted velocity
against each script to find the clip's global time offset — out of
sample, since the released model never saw your clip. The sidecar
records the fit, its confidence, a polarity check and a drift check.

A scripted clip is **admitted** to the training roster when the fit's
peak reaches 0.55, polarity is not suspect, and no drift alarm spreads
past one reversal tolerance (66.7 ms). `status <id>` shows each fact
beside its threshold. `rosters/holdout.json` is drawn once from the
admitted clips (one in eight) and never rewritten;
`rosters/train.json` is the admitted clips outside it, rewritten on
every `prepare`.

Curation stays per clip and per human: `status` to see why a clip is in
or out, `remove` to take it out. Nothing decides in bulk.
