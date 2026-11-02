# Reproducing the paper

Every table in the paper traces to a command here. The commands assume the repository root as the
working directory, `MOME_DATA` pointing at the data, and the ImageNet weight at
`./jx_vit_base_p16_224-80ecf9dd.pth`. A single RTX 5090 runs the Market-1501 and RegDB recipes;
VeRi-776 was trained on A100s.

```bash
export MOME_DATA=/path/to/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## What one deletion split costs

A split is not one training run. The retrain reference is a family of five, four of them are the
core's distillation teachers, and the calibrated attacks are scored against a second family that
shares no checkpoint with the first. Per split on Market-1501 that is ten retrains and one core.
The paper reports five splits, 42, 123, 456, 789 and 1024, on each of three benchmarks.

`scripts/repro/` runs a whole split end to end. Below is what it does, in the order it does it.

## 1. The reference families

```bash
SUB="MOME.FORGET_MODE 'sub_id' MOME.FORGET_ID_FRAC 0.5 MOME.SPLIT_SEED 42 MOME.DR_K 4"
CFG=configs/Market/vit_transreid_stride_mome_v3.yml

for s in 42 123 456 789 1024; do            # the reference family: Retrain row and the teachers
  python train_retrain_baseline.py --config_file $CFG \
    MODEL.DEVICE_ID "'0'" MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
    MOME.EVAL_TRAIN_MAP False $SUB SOLVER.SEED $s OUTPUT_DIR logs/sp42_market_kmix_s$s
done

for s in 2001 2002 2003 2004 2005; do        # the disjoint family the attacks are scored against
  ... same command, OUTPUT_DIR logs/sp42_market_xref_s$s
done
```

The two families must stay disjoint. An attack scored against a reference that shares a checkpoint
with the teachers reads the teachers, not the deletion.

## 2. The core

```bash
T=logs/sp42_market_kmix_s123/transformer_120.pth,logs/sp42_market_kmix_s456/transformer_120.pth,\
logs/sp42_market_kmix_s789/transformer_120.pth,logs/sp42_market_kmix_s1024/transformer_120.pth

python train_mome_v3.py --config_file $CFG \
  MODEL.DEVICE_ID "'0'" MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
  SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
  MOME.EVAL_TRAIN_MAP False $SUB SOLVER.SEED 42 \
  MODEL.PARASITIC_LINEAR True MODEL.PARASITIC_LORA_RANK 8 MODEL.PARASITIC_LORA_ALPHA 16 \
  MOME.TEACHER_WEIGHT "$T" MOME.TEACHER_CONFIG configs/Market/vit_transreid_stride.yml \
  MOME.CORE_KD 0.15 MOME.CORE_KD_TYPE "'feature'" \
  OUTPUT_DIR logs/sp42_market_core_s42
```

Per benchmark the recipe differs only in the schedule. VeRi-776 uses a core budget of eight passes
(`MOME.CORE_EVERY 4`), a 40-epoch cosine schedule and `MOME.CORE_KD 0.45` with three teachers.
RegDB uses an ensemble teacher of five retain-only AdamW retrains and
`scripts/run_with_regdb_cross.py` to drive the ten trials.

## 3. The Full payload

Consolidation folds the branch back in. It must run with label smoothing off:

```bash
python train_mome_v3.py --config_file $CFG ... MODEL.IF_LABELSMOOTH 'off' ...
```

`*_mome_v3.yml` does not set it, so the default in `config/defaults.py` turns it on and the
fine-tune then pulls the branch away from the model it has just reconstructed. On RegDB that one
flag moves Full from 49.8 to 66.6 mAP.

## 4. Reading the rows

```bash
# retrieval on the test split and on the deleted identities
python eval_partition.py --config_file $CFG --weight logs/sp42_market_core_s42/mome_v3_epoch_120.pth --mode core

# the three attacks, each against the disjoint reference family
python eval_advanced_mia.py --config_file $CFG --weight <ckpt> --mode core \
  --refs logs/sp42_market_xref_s2001,...,logs/sp42_market_xref_s2005

# Forgetting-Rank
python scripts/compute_rho.py --target <ckpt> --family logs/sp42_market_kmix_s42,...
```

`--mode core` and `--mode full` select the branch. Nothing else differs between the two readings:
the same file is loaded both times.

## 5. Across splits

```bash
for S in 42 123 456 789 1024; do bash scripts/repro/market_split.sh $S; done
python scripts/aggregate_splits.py          # writes logs/splitvar/aggregate.json
```

The tables report the mean and sample standard deviation over the five splits. On Market-1501 the
core beats the reference by 3.90 ± 0.48 mAP on the test split; on RegDB the margin on the deleted
identities is 10.34 ± 2.05.

## The baselines

Eleven post-hoc methods run from the same checkpoints and the same split:

```bash
python train_finetune_dr.py      # Fine-tune        python train_lotus.py     # LoTUS
python train_gradient_ascent.py  # GA               python train_munba.py     # MUNBa
python train_neggrad_plus.py     # NegGrad+         python train_loft.py      # LOFT
python train_salun.py            # SalUn            python train_lora_unlearn.py
python train_bad_teacher.py      # Bad Teacher      python unlearn_ssd.py     # SSD
python train_scrub.py            # SCRUB
```

Each takes `TEST.WEIGHT` for the model it adapts and writes into `OUTPUT_DIR`. LOFT fits a
projector; it needs `geoopt` installed, or it silently falls back to Euclidean Adam with a QR
retraction and the objective moves the other way.
