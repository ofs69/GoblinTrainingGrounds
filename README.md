# GoblinTrainingGrounds

Train, fine-tune, evaluate and export the models that
[GoblinScript](https://github.com/ofs69/GoblinScript) drafts funscripts with.

GoblinScript is the inference client: drop in a video, get a funscript. This
repository is where its models come from. It ships the two released models
(v0.5.1 and v0.6.0), the frozen perception they share, and one script that
takes a folder of your own video/funscript pairs to a model pack GoblinScript
can load.

## What you need

- A CUDA GPU with about 8 GB of memory. Training and perception run locally;
  nothing leaves your machine.
- Python 3.11.
- `ffmpeg` and `ffprobe` on your PATH. Any build from the last few years
  works. On Windows, `winget install --id Gyan.FFmpeg` and open a new
  terminal. On Linux, your package manager's `ffmpeg`. Check with:

  ```
  ffmpeg -version
  ffprobe -version
  ```

- Disk space for your project. Transcoded video and the latent caches are
  the bulk of it; the caches are the larger part and can be rebuilt.

## Setup

```
git clone --recurse-submodules https://github.com/ofs69/GoblinTrainingGrounds
cd GoblinTrainingGrounds
python -m venv venv
```

Activate the venv, on Windows:

```
venv\Scripts\activate
```

on Linux:

```
source venv/bin/activate
```

Then:

```
pip install -r requirements.txt
python goblintrain.py fetch
python goblintrain.py check
```

`fetch` downloads the two released checkpoints and verifies every file
against the hashes stored in `weights/`; the frozen encoder downloads
itself through torch.hub the first time something needs it (or now, with
`fetch --encoder`). `check` runs the decode-invariant check with no data
and confirms the install. Both are safe to run again at any time.

If you cloned without `--recurse-submodules`, run
`git submodule update --init` once.

## Your first project

A **project** is one directory that holds one dataset: your transcoded
clips, their scripts, the caches, the training runs and the drafts. It is
yours, it holds your media, and it must never go into git. Keep it outside
this repository or under `projects/`, which is ignored.

```
python goblintrain.py init projects/mine
```

Put the pairs you want to train on into a folder. Every video needs a
funscript with the same name beside it: `clip.mp4` and `clip.funscript`.
Pick them by hand; the quality of what you import is the quality of what
you train. Then:

```
python goblintrain.py import projects/mine picked/
```

The tool lists what it found before it touches anything: pairs, unpaired
files, duplicates of clips already in the project, unreadable media. It then
imports each pair on its own, transcoding it to the training grid and
sanitizing the script against the video's real duration, and prints the new
clip ID beside it. From here on a clip is its ID. Original names are kept
only in `private/map.tsv` inside the project, for you.

A refusal names its reason (no script, unreadable video, duplicate of clip
N, script longer than the video) and stops the import unless you pass
`--continue`. Pairs already imported stay imported.

```
python goblintrain.py prepare projects/mine
```

This is the long step: shot boundaries, latents, the script-to-video lag fit
with the released model as a reference, and admission into the training and
holdout rosters. It works clip by clip and only on what is missing, so you
can stop it at any time and run it again.

```
python goblintrain.py status projects/mine
```

One row per clip with every stage and its admission facts, and a summary of
how many clips train and how many are held out.

## Training

Fine-tune the released model on your clips:

```
python goblintrain.py train  projects/mine --from v0.6.0 --name mine
python goblintrain.py eval   projects/mine --run mine
python goblintrain.py export projects/mine --run mine --pack mine
```

`train` fits the deploy heads on the released model's frozen trunk against
your training roster. `eval` drafts your held-out clips and reads the
written funscripts; `--ref` prints another eval directory beside it for
comparison, the released model's for instance. `export` adds your pack to
a GoblinScript bundle and checks that the exported decode reproduces the
Python one on a clip of your project.

A trunk from scratch, for a corpus large enough to earn it:

```
python goblintrain.py train projects/mine --recipe recipes/v0.6.0.json --name fresh
```

The recipe holds every setting the released trunk used. A flag on the
command line overrides one recipe key. Every run records the settings it
resolved to in its own directory under `runs/`.

Draft a video with a released model (the default is v0.6.0), or with a
run of yours:

```
python goblintrain.py draft video.mp4
python goblintrain.py draft video.mp4 --model projects/mine/runs/mine
```

## Maintaining a project

**Adding clips.** Import the new pairs and run `prepare` again. Only the new
clips are processed. The rosters are rewritten to include them; existing
runs are untouched, and a new run trains on the new roster.

```
python goblintrain.py import  projects/mine more/
python goblintrain.py prepare projects/mine
```

**Removing clips.** `remove` moves the clip and everything derived from it
to the project's trash and rewrites the rosters. `--undo` puts the last
removal back. `purge` empties the trash for good.

```
python goblintrain.py remove projects/mine 000123
python goblintrain.py remove projects/mine --undo
python goblintrain.py purge  projects/mine
```

**Replacing a script.** A better funscript for a clip you already have is a
new pair: remove the old clip and import the video with the new script.
Duplicate detection looks at the script's content, so the new script is not
a duplicate.

**Duplicates.** Two imports of the same script are refused by content, even
from different files or folders. Two different scripts for the same video
both import; whether you want both is your call.

**Checking a project.** Every command starts by checking that the manifest
and the files agree and by clearing any transaction that did not finish. A
clip with files but no record, or a record without files, is reported and
never silently adopted. `status --verify` does the full check on demand,
including every cache's stamp against the frozen perception.

**Interrupting.** Stop any command with Ctrl-C. An import loses at most the
pair in flight; a `prepare` loses at most the stage in flight for one clip;
running the same `train` command again resumes from the last completed
epoch.
Nothing half-written is ever read back as if it were whole.

**Moving a project.** A project is one directory with relative paths inside.
Move or copy the whole directory anywhere, on any drive, and point the
commands at the new place.

**Backing up.** The whole project directory is the simple answer. If space
matters, `videos/`, `scripts/`, `meta/`, `manifest.jsonl` and `private/`
are the parts you cannot recreate; every cache and roster comes back from
`prepare`, and runs come back from training.

**Several projects.** Any number, anywhere. Nothing is shared between them
except the repository's code and weights.

**Do not** edit `manifest.jsonl`, the rosters or anything under the cache
directories by hand. Use the commands; they keep the project consistent.

## Updating

```
git pull --recurse-submodules
pip install -r requirements.txt
python goblintrain.py fetch
```

The perception is frozen, so your caches stay valid across updates. If an
update ever changed something a cache depends on, the tools would refuse
that cache and say so rather than reprocess anything on their own.

## Privacy

Everything runs on your machine. The tools refer to clips by ID and never
print an original filename or path. `private/map.tsv` is the only place the
original names live; it is inside your project, and nothing in this
repository reads it except `import`. If you work with a coding agent in this
tree, `AGENTS.md` tells it to stay out of your project's data.

## Layout

```
goblintrain.py      the one entry point (python goblintrain.py --help)
goblintrain/        the code behind it
recipes/            the settings the released models trained with
weights/            frozen perception, released checkpoints, their hashes
docs/               the model, the data format, training, evaluation, export
tests/              the smoke test and the transaction test
goblinscript/       the inference client, as a submodule
projects/           your projects, if you keep them here; ignored by git
```

## License

MIT. Third-party weights are listed in `THIRD-PARTY-NOTICES.md`.
