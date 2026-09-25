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

## What you need

- A CUDA GPU with approximately 8 GB of memory. Training and perception
  run locally. No data leaves your machine.
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

- Disk space for your project, on an SSD. `prepare` and training read the
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

- `fetch` downloads the two released checkpoints. It verifies each file
  against the hashes in `weights/`. The frozen encoder downloads through
  torch.hub the first time a command needs it. To download it now, run
  `fetch --encoder`.
- `check` runs the decode-invariant check without data and confirms the
  install.

You can run both commands again at any time.

If you cloned without `--recurse-submodules`, run
`git submodule update --init` one time.

## First project

A **project** is one directory that holds one dataset: your transcoded
clips, their scripts, the caches, the training runs and the drafts. A
project contains your media. Do not add it to git. Keep it outside this
repository or under `projects/`. Git ignores `projects/`.

```
python goblintrain.py init projects/mine
```

Put the training pairs into a folder. Each video needs a funscript with
the same name in the same folder: `clip.mp4` and `clip.funscript`. Select
the pairs by hand. Then:

```
python goblintrain.py import projects/mine picked/
```

Before it changes anything, `import` lists what it found: pairs, unpaired
files, duplicates of clips already in the project, unreadable media. Then
it imports each pair separately. For each pair, it:

1. transcodes the video to the training grid
2. sanitizes the script against the real duration of the video
3. prints the new clip ID.

After import, the ID identifies the clip. The original names are only in
`private/map.tsv` inside the project.

A refusal gives its reason: no script, unreadable video, duplicate of
clip N, or script longer than the video. A refusal stops the import. To
continue past refusals, pass `--continue`. Pairs that were already
imported stay imported.

```
python goblintrain.py prepare projects/mine
```

This is the longest step. It does:

- shot boundary detection
- latent extraction
- the script-to-video lag fit, with the released model as reference
- admission into the training and holdout rosters.

It processes one clip at a time and only the missing stages. You can stop
it at any time and run it again.

```
python goblintrain.py status projects/mine
```

`status` prints one row per clip with each stage and its admission facts.
Then it prints the number of training clips and the number of holdout
clips.

## Training

Fine-tune the released model on your clips:

```
python goblintrain.py train  projects/mine --from v0.6.0 --name mine
python goblintrain.py eval   projects/mine --run mine
python goblintrain.py export projects/mine --run mine --pack mine
```

- `train` fits the deploy heads on the frozen trunk of the released model,
  against your training roster.
- `eval` drafts your holdout clips and reads the written funscripts.
  `--ref` prints a second eval directory next to it for comparison, for
  example the eval of the released model.
- `export` adds your pack to a GoblinScript bundle. It checks that the
  exported decode gives the same result as the Python decode on a clip of
  your project.

Train a new trunk from scratch. Use this path for a large corpus:

```
python goblintrain.py train projects/mine --recipe recipes/v0.6.0.json --name fresh
```

The recipe contains every setting that the released trunk used. A
command-line flag overrides one recipe key. Each run records its resolved
settings in its directory under `runs/`.

Fine-tune the trunk, starting from the released trunk with a small
learning rate. Use this path when your clips show content that the
released model did not see:

```
python goblintrain.py train projects/mine --init-from v0.6.0 --name mine-trunk --set trunk.lr=3e-5 --set trunk.epochs=5
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
python goblintrain.py import  projects/mine more/
python goblintrain.py prepare projects/mine
```

**Remove clips.** `remove` moves the clip and all files derived from it to
the project trash. It rewrites the rosters. `--undo` restores the last
removal. `purge` deletes the trash permanently.

```
python goblintrain.py remove projects/mine 000123
python goblintrain.py remove projects/mine --undo
python goblintrain.py purge  projects/mine
```

**Replace a script.** A new funscript for an existing clip is a new pair.
Remove the old clip, then import the video with the new script. Duplicate
detection compares script content, so the new script is not a duplicate.

**Duplicates.** `import` refuses a second copy of the same script by
content, also from a different file or folder. Two different scripts for
the same video both import. You decide whether to keep both.

**Check a project.** Each command first checks that the manifest agrees
with the files. It also clears unfinished transactions. It reports a clip
with files but no record, or a record with no files. It never adopts them
automatically. `status --verify` runs the full check. The full check
includes the stamp of each cache against the frozen perception.

**Interrupt.** Press Ctrl-C to stop a command.

- `import` loses at most the pair in progress.
- `prepare` loses at most the stage in progress for one clip.
- `train` resumes from the last completed epoch when you run the same
  command again.

No command reads a partially written file as complete.

**Move a project.** A project is one directory with relative paths. Move
or copy the full directory to any location or drive. Then give the new
path to the commands.

**Back up.** Back up the full project directory. To save space, back up
only `videos/`, `scripts/`, `meta/`, `manifest.jsonl` and `private/`. You
cannot recreate these. `prepare` recreates every cache and roster.
Training recreates the runs.

**Multiple projects.** You can have any number of projects in any
location. Projects share only the code and weights of the repository.

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

All processing runs on your machine. The tools identify clips by ID. They
never print an original filename or path. `private/map.tsv` is the only
location of the original names. It is inside your project. In this
repository, only `import` reads it. If you use a coding agent in this
tree, `AGENTS.md` tells the agent to stay out of your project data.

## Layout

```
goblintrain.py      the one entry point (python goblintrain.py --help)
goblintrain/        the code behind it
recipes/            the settings the released models trained with
weights/            frozen perception, released checkpoints, their hashes
docs/               the model, the data format, training, evaluation, export
tests/              the smoke test and the transaction test
goblinscript/       the inference client, as a submodule
projects/           your projects, if you keep them here (ignored by git)
```

## License

MIT. `THIRD-PARTY-NOTICES.md` lists the third-party weights.
