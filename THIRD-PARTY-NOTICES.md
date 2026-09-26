# Third-party notices

The code in this repository and the released GoblinScript model
checkpoints (`v0.5.1.pt`, `v0.6.0.pt`, the bare trunks, the PCA basis and
`artifact_prior.json`) are ours and MIT-licensed (see `LICENSE`).

The pipeline uses third-party weights. This repository does not store
them. Each is downloaded from its own source and stays under its own
license:

- **V-JEPA 2.1 ViT-B video encoder** (Meta AI, `facebookresearch/vjepa2`).
  `fetch` downloads it through `torch.hub`. The weights belong to Meta
  and are distributed under the license in their repository. This repository does not mirror or modify
  them. The exported GoblinScript bundle embeds a traced graph of them,
  for local inference only.

- **TransNetV2 shot-boundary detector** (Tomáš Souček and Jakub Lokoč),
  through the `transnetv2_pytorch` pip package. MIT-licensed.

All other dependencies are ordinary pip packages (see `requirements.txt`).
This repository does not redistribute them.
