# Data

Nothing here is redistributed. Each dataset comes from its own authors under its own terms. Put
them under one directory and point the code at it, either by exporting `MOME_DATA` or by passing
`DATASETS.ROOT_DIR` on the command line. The configs default to `./data`.

```bash
export MOME_DATA=/path/to/data      # or: ... DATASETS.ROOT_DIR /path/to/data
```

## Layout

```
data/
  market1501/
    bounding_box_train/          # 12,936 images, 751 identities
    bounding_box_test/
    query/
  VeRi/
    image_train/                 # 37,778 images, 576 identities
    image_query/
    image_test/
  regdb_raw_kaggle/RegDB/
    Visible/ Thermal/            # 412 identities, 10 images each per modality
  regdb_cross/
    idx/                         # the 10-trial index, 40 files
  cifar-10-batches-py/           # only for the class-deletion and generality tables
  cifar-100-python/
  tiny-imagenet-200/
```

## Where each comes from

| Dataset | Source |
|:---|:---|
| Market-1501 | <https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html> |
| VeRi-776 | <https://vehiclereid.github.io/VeRi/> |
| RegDB | request from the authors of <https://doi.org/10.3390/s17030605> |
| CIFAR-10, CIFAR-100 | <https://www.cs.toronto.edu/~kriz/cifar.html>, downloaded by torchvision |
| TinyImageNet | <http://cs231n.stanford.edu/tiny-imagenet-200.zip> |

VeRi needs the viewpoint annotation that TransReID releases, `datasets/keypoint_train.txt` and
`datasets/keypoint_test.txt`. Both ship with this repository, and Side Information Embedding
reads them.

## RegDB, cross-modality

The cross-modality protocol follows DDAG and its ten trial splits. Build the index once:

```bash
python scripts/prepare_regdb_idx_cross.py
```

It writes `regdb_cross/idx/` from the index files that come with the RegDB release. Every
RegDB-cross command takes `--trial N` for `N` in 1 to 10 and `--test_mode v2t` or `t2v`.

## The ImageNet initialization

Every model starts from the ViT-B/16 weight TransReID uses, `jx_vit_base_p16_224-80ecf9dd.pth`.
It is not redistributed here. Download it from the TransReID repository or from timm and put it at
the repository root, or point `MODEL.PRETRAIN_PATH` at it.

## How a deletion request is formed

`datasets/make_dataloader_mome.py` splits the training identities, not the images. With
`MOME.FORGET_ID_FRAC 0.5` half the identities are deleted; `MOME.DR_K 4` keeps four images of each
retained identity, which is the operational case of a controller who must keep a thin gallery.
`MOME.SPLIT_SEED` picks the split, and the paper reports five of them, 42, 123, 456, 789 and 1024.
On Market-1501 that is 376 deleted identities, 6,473 images in the forget set and 1,491 in the
retain set.
