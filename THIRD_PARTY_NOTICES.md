# Third-party notices

This repository builds on code, weights and data released by others. Each item below keeps the
terms of its own release.

## Code

**TransReID** — <https://github.com/heshuting555/TransReID>, MIT License, Copyright (c) 2021
heshuting555. `config/`, `datasets/`, `model/`, `loss/`, `solver/`, `processor/`, `utils/`,
`train.py`, `test.py` and `configs/` derive from it. The parasitic parameters, gradient isolation,
the two-step loop and everything under `scripts/` are this work; the files they live in are marked
where they were modified. A copy of the MIT License is in [LICENSE](LICENSE), which covers both
the upstream code and this work.

**timm** — <https://github.com/huggingface/pytorch-image-models>, Apache License 2.0. The Vision
Transformer in `model/backbones/vit_pytorch.py` follows timm's implementation, as it does upstream.

**The unlearning baselines** are reimplementations written against this codebase, from the
descriptions and the released code of their papers: Fine-tune, Gradient Ascent, NegGrad+, SalUn,
Bad Teacher, SCRUB, SSD, LoTUS, MUNBa, LOFT and a LoRA adapter. Each entry point names its source
in its docstring. We thank those authors for releasing their work.

## Pre-trained weights

**ViT-B/16, ImageNet-21k** — `jx_vit_base_p16_224-80ecf9dd.pth`, the initialization every model
here starts from, is the timm weight TransReID uses. It is not redistributed in this repository;
see [docs/DATA.md](docs/DATA.md) for where to get it.

## Data

None of the datasets are redistributed here. Each is obtained from its own authors under its own
terms: **Market-1501**, **VeRi-776**, **RegDB**, **CIFAR-10**, **CIFAR-100** and **TinyImageNet**.
[docs/DATA.md](docs/DATA.md) says where each comes from and how to lay it out.

RegDB is used under the cross-modality protocol of **DDAG** (<https://github.com/mangye16/DDAG>)
and the ten trial index files released with it.
