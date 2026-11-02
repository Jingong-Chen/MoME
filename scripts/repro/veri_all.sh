# One VeRi-776 deletion split: the two retrain families, the core, the Core-only control and the
# consolidated Full payload. Copied verbatim from repro/zlt_families.sh and the train_veri() of
# repro/xref_band.sh, with MOME.SPLIT_SEED as the only change.
#
# The reference structure differs from Market. Per the audit the VeRi attack references are
# kmix_s{789,1024} plus xref_s{2001,2002}, the band adds xref_s2003, and the teachers the core
# distills from are kmix_s{42,123,456}, three rather than five because five frozen teachers plus
# the student do not fit in 32 GB. So eight retrains per split, not ten.
#
# The eleven post-hoc baseline rows are NOT here. Their training logs are not in the repository,
# only their final checkpoints, so the hyperparameters are not recoverable and guessing them is
# what produced the wrong RegDB Fine-tune row.
#
#   usage: bash repro/splitvar/veri_all.sh <split_seed>
#!/usr/bin/env bash
# Portable form of the script that produced the paper's numbers. It assumes the repository root as
# the working directory and MOME_DATA pointing at the datasets; everything else is as it was run.
set -o pipefail
: "${MOME_DATA:=./data}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD"
LOGDIR=${LOGDIR:-logs/_repro}
mkdir -p "$LOGDIR"
S=$1
[ -z "$S" ] && { echo "need a split seed"; exit 2; }

P=sp${S}_veri
CFG=configs/VeRi/vit_transreid_stride_mome_v3.yml
SUB=(MOME.FORGET_MODE "'sub_id'" MOME.FORGET_ID_FRAC 0.5 MOME.SPLIT_SEED $S MOME.DR_K 4)
LORA=(MODEL.PARASITIC_LINEAR True MODEL.PARASITIC_LORA_RANK 8 MODEL.PARASITIC_LORA_ALPHA 16)
FC=("${LORA[@]}" MODEL.PARASITIC_LINEAR_MODE "'full'" MODEL.PARASITIC_EMBED True
    MODEL.PARASITIC_HEAD_ON True MODEL.PARASITIC_BN_STATS True MODEL.IF_LABELSMOOTH "'off'")
BASE=(MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA"
      MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth)

retrain () { # tag seed
  local out=logs/$1
  [ -f $out/transformer_120.pth ] && { echo "have $1"; return 0; }
  python -u train_retrain_baseline.py --config_file $CFG "${BASE[@]}" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
    MOME.EVAL_TRAIN_MAP False "${SUB[@]}" SOLVER.SEED $2 OUTPUT_DIR $out \
    > "$LOGDIR/$1.log" 2>&1 || { echo "$1_FAILED"; return 1; }
  echo "$1 done"
}
echo "===== split $S : retrain families ====="
for s in 42 123 456 789 1024; do retrain ${P}_kmix_s$s $s || exit 1; done
for s in 2001 2002 2003;        do retrain ${P}_xref_s$s $s || exit 1; done

VT=logs/${P}_kmix_s42/transformer_120.pth,logs/${P}_kmix_s123/transformer_120.pth,logs/${P}_kmix_s456/transformer_120.pth

# The core needs the card to itself: the student plus three frozen teachers leaves very little
# headroom, and a 3 GB leftover from the previous stage was enough to kill the first attempt.
wait_for_gpu () {
  for i in $(seq 1 120); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -ge 28000 ] && return 0
    echo "waiting for the card, ${free} MiB free"; sleep 30
  done
  echo "the card never freed up"; return 1
}

core () { # outdir extra...
  local out=$1; shift
  wait_for_gpu || return 1
  [ -f $out/mome_v3_epoch_40.pth ] && { echo "have $(basename $out)"; return 0; }
  python -u train_mome_v3.py --config_file $CFG "${BASE[@]}" \
    SOLVER.MAX_EPOCHS 40 SOLVER.EVAL_PERIOD 40 SOLVER.CHECKPOINT_PERIOD 40 \
    MOME.EVAL_TRAIN_MAP False "${SUB[@]}" SOLVER.SEED 42 "${LORA[@]}" MOME.CORE_EVERY 4 \
    MOME.TEACHER_WEIGHT "$VT" MOME.TEACHER_CONFIG configs/VeRi/vit_transreid_stride.yml \
    MOME.CORE_KD 0.45 MOME.CORE_KD_TYPE "'feature'" "$@" OUTPUT_DIR $out \
    > "$LOGDIR/$(basename $out).log" 2>&1 || { echo "$(basename $out)_FAILED"; return 1; }
  echo "$(basename $out) done"
}
echo "===== split $S : core and the Core-only control ====="
core logs/${P}_core_s42 || exit 1
core logs/${P}_coreonly MOME.CORE_ONLY True || exit 1

echo "===== split $S : consolidated Full payload ====="
if [ ! -f logs/${P}_full.pth ]; then
  python -u scripts/cvpr2027/full_fit.py --config_file $CFG \
    --weight logs/${P}_core_s42/mome_v3_epoch_40.pth \
    --orig logs/veri776_standard/transformer_120.pth \
    --init svd --epochs 30 --lr 0.004 --out logs/${P}_full.pth -- \
    "${BASE[@]}" "${SUB[@]}" "${FC[@]}" > "$LOGDIR/${P}_full.log" 2>&1 \
    || { echo "${P}_FULL_FAILED"; exit 1; }
fi
echo "[ok] full  $(grep -E '^DONE' "$LOGDIR/${P}_full.log" | tail -1)"
echo "VERI_SPLIT_${S}_COMPLETE"
