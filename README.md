# GoblinTrainingGrounds

Train, fine-tune, evaluate and export the models that
[GoblinScript](https://github.com/ofs69/GoblinScript) uses to draft
funscripts.

GoblinScript is the inference client. It takes a video and writes a
funscript. This repository produces its models. It contains:

- the two released models (v0.5.1 and v0.6.0)
- the frozen perception that both models use
- one script that turns a folder of your video/funscript pairs into a
  model pack that GoblinScript can load.

## Do you need to train?

To get drafts, you do not need to train. GoblinScript and
`goblintrain draft` draft with v0.6.0.

The size of a training set is measured in hours of scripted footage, not
in number of clips. v0.6.0 trained on approximately 105 hours of scripted
footage. A model that you train here learns its heads from your footage
alone. With much less footage, it drafts worse than v0.6.0. A test with
21 minutes was worse on every metric, also on the footage it trained on.
The amount of footage at which a model of your own is better than v0.6.0
is not measured. For a custom dataset, start with 5 to 10 hours of
scripted footage. `status` prints the video length of your scripted
clips. For v0.6.0, 90% of the video length was scripted.

Train only if you have a dataset of that size. Then keep your model only
if its eval on the holdout is better than the eval of v0.6.0 (see
[Training](#training)).

## What you need

- A CUDA GPU with approximately 8 GB of memory. Training and perception
  run locally. No data leaves your machine.
  - AMD GPU: not tested. The code uses only standard PyTorch, and the
    ROCm build of PyTorch uses the same `cuda` device name. Thus it can
    work on Linux. Install the ROCm builds of `torch` and `torchvision`
    in place of the `+cu126` builds in `requirements.txt`. Use an RDNA3
    or newer card, because training uses bf16. `export` uses DirectML to
    check the encoder graph and to run parity. DirectML is available only
    on Windows. Thus train and eval on Linux, then export on Windows.
  - GoblinScript drafts on any GPU through DirectML. Only this
    repository needs the PyTorch GPU.
- Python 3.11.
- `ffmpeg` and `ffprobe` on your PATH. Any build from the last few years
  works.
  - Windows: run `winget install --id Gyan.FFmpeg`, then open a new
    terminal.
  - Linux: install `ffmpeg` from your package manager.

  Check the install:

  ```
  ffmpeg -version
  ffprobe -version
  ```

- Disk space for your projects, on an SSD. Projects are inside the
  repository, so clone it onto the SSD. `prepare` and training read the
  latent caches continuously. On a spinning disk or a network drive, the
  disk limits the speed, not the GPU. Transcoded video and latent caches
  use most of the space. The caches are the larger part, and they can be
  rebuilt.

## Setup

```
git clone --recurse-submodules https://github.com/ofs69/GoblinTrainingGrounds
cd GoblinTrainingGrounds
python -m venv venv
```

Activate the venv on Windows:

```
venv\Scripts\activate
```

Activate the venv on Linux:

```
source venv/bin/activate
```

Then:

```
pip install -r requirements.txt
python goblintrain.py fetch
python goblintrain.py check
```

- `fetch` downloads the two released checkpoints and their bare trunks
  (for `train --from`). It verifies each file against the hashes in
  `weights/`. Then it downloads the frozen encoder (1.7 GB) through
  torch.hub into the torch.hub cache. If the cache already has the
  encoder, `fetch` does not download it again.
- `check` runs the decode-invariant check without data and confirms the
  install. It runs on the CPU.

You can run both commands again at any time.

Confirm that PyTorch sees your GPU. This prints `True`:

```
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, `prepare`, `train` and `eval` run on the CPU, which
is very slow. Then install the GPU build of `torch` from
`requirements.txt` again.

If you cloned without `--recurse-submodules`, run
`git submodule update --init` one time.

## Use a coding agent

We recommend that you do the work with a coding agent, for example Claude
Code or Codex, opened in the repository root. The stages take hours, and
their output is long: the eval prints five tables and a paired
bootstrap. An agent can:

- run the stages and resume them after an interruption
- read the eval tables and tell you if your run beats v0.6.0
- explain an error and fix its cause.

Tell it the goal, for example: "Prepare project mine, train on it, and
tell me if the run drafts better than v0.6.0 on the holdout."

The agent reads `AGENTS.md`. It tells the agent to:

- identify clips only by ID, never by the original name
- stay out of `private/`, and not list the media and cache directories
- not send media, scripts, latents or checkpoints off the machine
- leave to you: `import`, removing clips, changing a recipe and
  publishing.

Run `import` yourself, because its listing shows the original file
names. A cloud agent sends what it reads to its provider: command output,
logs, clip IDs and metrics. `AGENTS.md` gives instructions, but it does
not enforce them. To enforce them, deny the agent read access to `projects/*/private/`, `projects/*/videos/`
and `projects/*/scripts/` in its permission settings.

## First project

A **project** is one directory that holds one dataset: your transcoded
clips, their scripts, the caches, the training runs and the drafts. Each
project has a name. The commands take the name, and the project is the
directory `projects/<name>` in this repository. A path is not accepted.
A project contains your media. Git ignores `projects/`.

```
python goblintrain.py init mine
```

Put the training pairs into a folder. Each video needs a funscript with
the same name in the same folder: `clip.mp4` and `clip.funscript`. Select
the pairs by hand. A clip trains only if it is at least as long as the
training window of the recipe: 3 min 25 s for the shipped recipes. A
shorter clip can be evaluated and drafted. `import` marks each pair that
is too short to train. Then:

```
python goblintrain.py import mine picked/
```

Before it changes anything, `import` lists what it found: pairs, unpaired
files, duplicates of clips already in the project, unreadable media. Then
it imports each pair separately. For each pair, it:

1. transcodes the video to the training grid
2. sanitizes the script
3. prints the new clip ID.

Sanitization cuts the script at the end of the video. It also lowers
strokes that are faster than a device can play (600 position units per
second). A lowered stroke keeps its timing and loses depth. `docs/DATA.md`
gives the full rules.

After import, the ID identifies the clip. The original names are only in
`private/map.tsv` inside the project.

A refusal gives its reason: unreadable file, a script that is not
funscript JSON, a video that ffprobe cannot read, duplicate of clip N, or
fewer than two usable actions. A refusal stops the import. To continue
past refusals, pass `--continue`. Pairs that were already imported stay
imported.

Actions past the end of the video are cut off. If the script runs more
than 1 s past the end, the listing shows a warning, because the script
may be for another cut of the video. The pair is still imported.

```
python goblintrain.py prepare mine
```

This is the longest step. It does:

- shot boundary detection
- latent extraction
- admission into the training and holdout rosters.

It processes one clip at a time and only the missing stages. You can stop
it at any time and run it again.

The holdout is every eighth admitted clip. Training never sees it, and
eval compares models on it. The first `prepare` of all clips draws it,
and it never changes after that. A `prepare` of some clip IDs does not
draw it. Thus import all your clips, at least 8, before the first
`prepare` of all clips. With fewer, the holdout is empty and stays
empty, and you cannot compare a run with v0.6.0.

`prepare` expects each script to be synchronized with its video. If your
scripts can be offset from their videos, add `--lag-fit`. Then `prepare`
also measures the offset of each script against the released model.
Training and eval shift the script by that offset, and a clip with an
uncertain fit is not admitted. See `docs/DATA.md`, "Lag fit and
admission".

```
python goblintrain.py status mine
```

`status` prints one row per clip with:

- its length and its action count
- whether `prepare` has processed it
- its roster
- the reason if it is not admitted.

Then it prints the total video length and the size of the training
roster and the holdout.

## Training

Read [Do you need to train?](#do-you-need-to-train) first. A small
project gives a model that is worse than v0.6.0.

Train new heads on the frozen trunk of v0.6.0, then compare the result
with v0.6.0 on the holdout:

```
python goblintrain.py eval   mine --run v0.6.0
python goblintrain.py train  mine --from v0.6.0 --name mine
python goblintrain.py eval   mine --run mine --ref v0.6.0-holdout
```

- The first `eval` drafts your holdout clips with v0.6.0 and reads the
  written funscripts. This is the baseline.
- `train` keeps the trunk of v0.6.0 and trains the deploy heads from a
  random init on your training roster. It does not start from the heads of
  v0.6.0.
- The second `eval` drafts the same clips with your run. `--ref` prints
  the baseline next to it, with paired bootstrap intervals.
  `docs/EVALUATION.md` tells how to read the tables.

Keep your run only if it is better than v0.6.0 on the holdout, in the
tails too: worst clips, spikes per minute. If it is not, use v0.6.0. If it
is, export it:

```
python goblintrain.py export mine --run mine --pack mine
```

`export` adds your pack to a GoblinScript bundle. It checks that the
exported decode gives the same result as the Python decode on a clip of
your project.

Train a new trunk from a random init. Use this path only for a dataset
that is larger than the approximately 105 scripted hours of v0.6.0:

```
python goblintrain.py train mine --recipe recipes/v0.6.0.json --name fresh
```

The recipe contains every setting that the released trunk used. A
command-line flag overrides one recipe key. Each run records its resolved
settings in its directory under `runs/`.

Fine-tune the trunk, starting from the released trunk with a small
learning rate. Then the heads train from a random init as with `--from`.
Thus the same size rule applies. On the training data of v0.6.0, this
path gave no gain. A gain can come only from many new clips:

```
python goblintrain.py train mine --init-from v0.6.0 --name mine-trunk --set trunk.lr=3e-5 --set trunk.epochs=5
```

Draft a video with a released model (default: v0.6.0) or with one of your
runs:

```
python goblintrain.py draft video.mp4
python goblintrain.py draft video.mp4 --model projects/mine/runs/mine
```

## Project maintenance

**Add clips.** Import the new pairs and run `prepare` again. `prepare`
processes only the new clips. It rewrites the rosters to include them.
Existing runs do not change. A new run trains on the new roster.

```
python goblintrain.py import  mine more/
python goblintrain.py prepare mine
```

**Remove clips.** `remove` moves the clip and all files derived from it to
the project trash. The clip leaves every roster, including the holdout.
`--undo` restores the last removal and puts the clip back in its rosters.
`purge` deletes the trash permanently. An ID is never given to another
clip, even after `purge`.

```
python goblintrain.py remove mine 000123
python goblintrain.py remove mine --undo
python goblintrain.py purge  mine
```

**Replace a script.** A new funscript for an existing clip is a new pair.
Remove the old clip, then import the video with the new script. Duplicate
detection compares script content, so the new script is not a duplicate.

**Duplicates.** `import` refuses a second copy of the same script by
content, also from a different file or folder. Two different scripts for
the same video both import. You decide whether to keep both.

**Check a project.** Each command first clears unfinished transactions.
`status` checks that the manifest agrees with the files. It reports a
clip with files but no record, or a record with no files. It never adopts
them automatically. `status --verify` also checks the stamp of each latent
cache against the frozen perception. A command that reads a cache with a
wrong stamp refuses it.

**Interrupt.** Press Ctrl-C to stop a command.

- `import` loses at most the pair in progress.
- `prepare` loses at most the stage in progress for one clip.
- `train` resumes from the last completed epoch when you run the same
  command again.

No command reads a partially written file as complete.

**Move or rename a project.** A project is one directory with relative
paths. Move or copy the full directory into `projects/` of any clone of
this repository. The directory name is the project name.

**Back up.** Back up the full project directory. To save space, back up
only `videos/`, `scripts/`, `meta/`, `manifest.jsonl` and `private/`. You
cannot recreate these. `prepare` recreates every cache and roster.
Training recreates the runs.

**Multiple projects.** You can have any number of projects in
`projects/`. Projects share only the code and weights of the repository.

**Do not** edit `manifest.jsonl`, the rosters or files in the cache
directories by hand. Use the commands. The commands keep the project
consistent.

## Updates

```
git pull --recurse-submodules
pip install -r requirements.txt
python goblintrain.py fetch
```

The perception is frozen, so your caches stay valid after an update. If an
update changes something that a cache depends on, the tools refuse that
cache and tell you. They do not reprocess anything automatically.

## Privacy

All processing runs on your machine. The tools identify clips by ID.
Only the `import` listing shows the names of the files you import.
`private/map.tsv` is the only stored location of the original names. It
is inside your project. In this repository, only `import`, `remove` and
`status` read it. If you use a coding agent in this tree, `AGENTS.md`
tells the agent to stay out of your project data. See
[Use a coding agent](#use-a-coding-agent).

## Layout

```
goblintrain.py      the one entry point (python goblintrain.py --help)
goblintrain/        the code behind it
recipes/            the settings the released models trained with
weights/            frozen perception, released checkpoints, their hashes
docs/               the model, the data format, training, evaluation, export
tests/              the smoke test and the transaction test
goblinscript/       the inference client, as a submodule
projects/           your projects, one directory each (ignored by git)
```

## License

MIT. `THIRD-PARTY-NOTICES.md` lists the third-party weights.
