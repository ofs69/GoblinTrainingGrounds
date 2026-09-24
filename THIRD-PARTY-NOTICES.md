# Third-party notices

The code in this repository and the released GoblinScript model
checkpoints (`v0.5.1.pt`, `v0.6.0.pt`, the bare trunks, the PCA basis and
`artifact_prior.json`) are ours and MIT-licensed (see `LICENSE`).

The pipeline builds on weights that are not ours. None of them are stored
in this repository; each is fetched from its own source and remains under
its own license:

- **V-JEPA 2 ViT-B video encoder** (Meta AI, `facebookresearch/vjepa2`).
  Downloaded through `torch.hub` on first use (`fetch --encoder` fetches
  it eagerly). The weights are Meta's and are distributed under the
  license published in their repository; this repository does not mirror
  or modify them, and the exported GoblinScript bundle embeds a traced
  graph of them for local inference only.

- **TransNetV2 shot-boundary detector** (Tomáš Souček and Jakub Lokoč),
  via the `transnetv2_pytorch` pip package. MIT-licensed.

Everything else arrives as an ordinary pip dependency (see
`requirements.txt`) and is not redistributed here.
