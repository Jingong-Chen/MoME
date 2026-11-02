# One deletion split of Market-1501, end to end, so Tables 1 and 2 can carry a spread over splits
# rather than over training seeds. Everything below is copied verbatim from the recipes that
# produced the deployed split, with MOME.SPLIT_SEED as the only change.
#
#   usage: bash repro/splitvar/market_split.sh <split_seed>
#
# Per split this trains ten retrains and one core:
#   kmix_s{42,123,456,789,1024}  the reference family; its five-seed mean is the Retrain row,
#                                and four of its members are the core's distillation teachers
#   xref_s{2001..2005}           the disjoint family the calibrated attacks are scored against
#   core_s42                     the deployed core, at training seed 42 on every split
#!/usr/bin/env bash
# Portable form of the script that produced the paper's numbers. It assumes the repository root as
# the working directory and MOME_DATA pointing at the datasets; everything else is as it was run.
set -o pipefail
: "${MOME_DATA:=./data}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD"
LOGDIR=${LOGDIR:-logs/_repro}
mkdir -p "$LOGDIR"
SPLIT=$1
[ -z "$SPLIT" ] && { echo "need a split seed"; exit 2; }

SUB=(MOME.FORGET_MODE "'sub_id'" MOME.FORGET_ID_FRAC 0.5 MOME.SPLIT_SEED $SPLIT MOME.DR_K 4)
LORA=(MODEL.PARASITIC_LINEAR True MODEL.PARASITIC_LORA_RANK 8 MODEL.PARASITIC_LORA_ALPHA 16)
CFG=configs/Market/vit_transreid_stride_mome_v3.yml
P=sp${SPLIT}_market

retrain () {  # tag seed
  local out=logs/$1 seed=$2
  [ -f $out/transformer_120.pth ] && { echo "have $1"; return 0; }
  python -u train_retrain_baseline.py --config_file $CFG \
    MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
    MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
    MOME.EVAL_TRAIN_MAP False "${SUB[@]}" SOLVER.SEED $seed OUTPUT_DIR $out \
    > "$LOGDIR/$1.log" 2>&1 || { echo "$1_FAILED"; return 1; }
  echo "$1 done"
}

for s in 42 123 456 789 1024; do retrain ${P}_kmix_s$s $s || exit 1; done
for s in 2001 2002 2003 2004 2005; do retrain ${P}_xref_s$s $s || exit 1; done
echo "${P}_RETRAINS_DONE"

MT=logs/${P}_kmix_s123/transformer_120.pth,logs/${P}_kmix_s456/transformer_120.pth,logs/${P}_kmix_s789/transformer_120.pth,logs/${P}_kmix_s1024/transformer_120.pth
if [ ! -f logs/${P}_core_s42/mome_v3_epoch_120.pth ]; then
  python -u train_mome_v3.py --config_file $CFG \
    MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
    MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
    MOME.EVAL_TRAIN_MAP False "${SUB[@]}" SOLVER.SEED 42 "${LORA[@]}" \
    MOME.TEACHER_WEIGHT "$MT" MOME.TEACHER_CONFIG configs/Market/vit_transreid_stride.yml \
    MOME.CORE_KD 0.15 MOME.CORE_KD_TYPE "'feature'" \
    OUTPUT_DIR logs/${P}_core_s42 > "$LOGDIR/${P}_core_s42.log" 2>&1 \
    || { echo "${P}_CORE_FAILED"; exit 1; }
fi
echo "${P}_CORE_DONE"
