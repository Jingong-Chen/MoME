# Everything Table 2 needs on one RegDB deletion split. RegDB runs through the cross-modality
# wrapper and its retrains use AdamW, since the SGD schedule used elsewhere does not train at all
# on the 824 retained images. Hyperparameters are read off the train_log.txt of the deployed runs.
#
#   usage: bash repro/splitvar/regdb_all.sh <split_seed>
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

P=sp${S}_regdb
CFG=configs/RegDB/vit_transreid_stride_cross_mome_v3.yml
ORIG=logs/regdb_cross_standard/trial_1/transformer_120.pth
SUB=(MOME.FORGET_MODE "'sub_id'" MOME.FORGET_ID_FRAC 0.5 MOME.SPLIT_SEED $S MOME.DR_K 4
     MOME.DR_K_STRATIFY_CAM True)
LORA=(MODEL.PARASITIC_LINEAR True MODEL.PARASITIC_LORA_RANK 8 MODEL.PARASITIC_LORA_ALPHA 16)
FC=("${LORA[@]}" MODEL.PARASITIC_LINEAR_MODE "'full'" MODEL.PARASITIC_EMBED True
    MODEL.PARASITIC_HEAD_ON True MODEL.PARASITIC_BN_STATS True MODEL.IF_LABELSMOOTH "'off'")
BASE=(MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA"
      MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth MODEL.SIE_CAMERA False)
X () { python -u scripts/run_with_regdb_cross.py --script "$1" --trial 1 --test_mode v2t -- "${@:2}"; }

retrain () { # tag seed
  local out=logs/$1
  [ -f $out/transformer_120.pth ] && { echo "have $1"; return 0; }
  X train_retrain_baseline.py --config_file $CFG "${BASE[@]}" \
    SOLVER.OPTIMIZER_NAME "'AdamW'" SOLVER.BASE_LR 0.0003 SOLVER.WARMUP_EPOCHS 3 \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
    MOME.EVAL_TRAIN_MAP False "${SUB[@]}" SOLVER.SEED $2 OUTPUT_DIR $out \
    > "$LOGDIR/$1.log" 2>&1 || { echo "$1_FAILED"; return 1; }
  echo "$1 done"
}

echo "===== split $S : retrain families ====="
for s in 42 123 456 789 1024; do retrain ${P}_adamw_s$s $s || exit 1; done
for s in 2001 2002 2003 2004 2005; do retrain ${P}_xref_s$s $s || exit 1; done

T=logs/${P}_adamw_s42/transformer_120.pth,logs/${P}_adamw_s123/transformer_120.pth,logs/${P}_adamw_s456/transformer_120.pth,logs/${P}_adamw_s789/transformer_120.pth,logs/${P}_adamw_s1024/transformer_120.pth

core () { # outdir extra...
  local out=$1; shift
  [ -f $out/mome_v3_epoch_120.pth ] && { echo "have $(basename $out)"; return 0; }
  X train_mome_v3_cross.py --config_file $CFG "${BASE[@]}" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 \
    MOME.EVAL_TRAIN_MAP False "${SUB[@]}" "${LORA[@]}" SOLVER.SEED 42 \
    MOME.TEACHER_WEIGHT "$T" MOME.TEACHER_CONFIG $CFG \
    MOME.CORE_KD 0.15 MOME.CORE_KD_TYPE "'feature'" "$@" OUTPUT_DIR $out \
    > "$LOGDIR/$(basename $out).log" 2>&1 || { echo "$(basename $out)_FAILED"; return 1; }
  echo "$(basename $out) done"
}
echo "===== split $S : core and the Core-only control ====="
core logs/${P}_core_s42 || exit 1
core logs/${P}_coreonly MOME.CORE_ONLY True || exit 1

echo "===== split $S : consolidated Full payload ====="
if [ ! -f logs/${P}_full.pth ]; then
  X scripts/cvpr2027/full_fit.py --config_file $CFG \
    --weight logs/${P}_core_s42/mome_v3_epoch_120.pth --orig $ORIG \
    --init svd --epochs 30 --lr 0.004 --out logs/${P}_full.pth -- \
    "${BASE[@]}" "${SUB[@]}" "${FC[@]}" > "$LOGDIR/${P}_full.log" 2>&1 \
    || { echo "${P}_FULL_FAILED"; exit 1; }
fi
echo "[ok] full  $(grep -E '^DONE' "$LOGDIR/${P}_full.log" | tail -1)"

echo "===== split $S : post-hoc baseline rows ====="
run () { # tag sentinel script lr args...   sentinel "latest" means any transformer_*.pth
  local tag=$1 sentinel=$2 script=$3 lr=$4; shift 4
  local out=logs/${P}_$tag
  if [ "$sentinel" = latest ]; then
    [ -n "$(ls $out/transformer_*.pth 2>/dev/null)" ] && { echo "have $tag"; return 0; }
  else
    [ -f "$out/$sentinel" ] && { echo "have $tag"; return 0; }
  fi
  X "$script" --config_file $CFG "$@" "${BASE[@]}" "${SUB[@]}" \
    SOLVER.SEED 42 SOLVER.BASE_LR $lr TEST.WEIGHT $ORIG OUTPUT_DIR $out \
    > "$LOGDIR/${P}_$tag.log" 2>&1 || { echo "[FAIL] $tag"; return 1; }
  echo "[ok] $tag"
}
# Fine-tune on RegDB runs AdamW at 3e-4 for 30 epochs, the recipe behind the published row
# (54.8 v2t mAP). Under the shared SGD schedule it collapses to 5.2 and was rejected.
run finetune      latest                    train_finetune_dr.py     0.0003 SOLVER.OPTIMIZER_NAME "'AdamW'" SOLVER.MAX_EPOCHS 30 SOLVER.WARMUP_EPOCHS 3
run ga            gradient_ascent_final.pth train_gradient_ascent.py 0.008 --ascent_epochs 1 --finetune_epochs 5
run salun         salun_final.pth           train_salun.py           0.0005 --epochs 10 --sparsity 0.5 --lambda_random 0.5
run ssd           ssd_unlearned.pth         unlearn_ssd.py           0.008 --alpha 0.7 --lambda_thr 10.0
run bad_teacher   bad_teacher_final.pth     train_bad_teacher.py     0.0005 --epochs 20 --lambda_ce 0.5
run lotus         lotus_final.pth           train_lotus.py           0.008 --epochs 5 --lr 1e-5 --alpha 2.0 --subset_size 0.3
run munba         munba_final.pth           train_munba.py           0.008 --epochs 10 --lr 0.01 --momentum 0.9 --weight_decay 0.0005
run loft          loft_U.pt                 train_loft.py            0.008 --steps 100 --lr 1.0 --weight_decay 0.05 --energy 0.95
run lora          lora_unlearn_final.pth    train_lora_unlearn.py    0.008 --epochs 20 --rank 4 --alpha 8
if [ -f logs/${P}_lora/lora_unlearn_final.pth ] && [ ! -f logs/${P}_lora/lora_merged.pth ]; then
  python -u scripts/rebuttal/merge_and_eval_lora.py --in_ckpt logs/${P}_lora/lora_unlearn_final.pth \
    --out_ckpt logs/${P}_lora/lora_merged.pth > "$LOGDIR/${P}_lora_merge.log" 2>&1 \
    && echo "[ok] lora merge" || echo "[FAIL] lora merge"
fi
run scrub_00005   scrub_final.pth           train_scrub.py           0.0005 --rounds 5 --max_epochs 2 --min_epochs 3 --alpha 1.0 --beta 1.0 --gamma 1.0
run neggrad_0008  neggrad_plus_final.pth    train_neggrad_plus.py    0.0008 --epochs 20 --alpha 1.0 --beta 0.5
echo "REGDB_SPLIT_${S}_COMPLETE"
