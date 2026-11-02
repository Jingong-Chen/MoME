# The eleven post-hoc rows of Table 1 on one Market deletion split.
#
# Every method starts from the full-data original, which does not depend on the split and is
# therefore reused, and every hyperparameter below is read off the train_log.txt of the run that
# produced the deployed row, so these are the same methods at the same settings on new identities.
#
#   usage: bash repro/splitvar/market_baselines.sh <split_seed>
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

CFG=configs/Market/vit_transreid_stride_mome_v3.yml
ORIG=logs/kmix_market_original/transformer_120.pth
P=sp${S}_market
COMMON=(MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA"
        MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth
        MOME.FORGET_MODE "'sub_id'" MOME.FORGET_ID_FRAC 0.5 MOME.SPLIT_SEED $S MOME.DR_K 4
        SOLVER.SEED 42 TEST.WEIGHT $ORIG)

run () { # tag sentinel script args... -- yacs...
  local tag=$1 sentinel=$2 script=$3; shift 3
  local out=logs/${P}_$tag
  [ -f "$out/$sentinel" ] && { echo "have $tag"; return 0; }
  echo "----- $tag $(date +%H:%M) -----"
  python -u "$script" --config_file $CFG "$@" "${COMMON[@]}" OUTPUT_DIR $out \
    > "$LOGDIR/${P}_$tag.log" 2>&1 || { echo "[FAIL] $tag"; return 1; }
  echo "[ok] $tag"
}

run finetune    transformer_120.pth    train_finetune_dr.py
run ga          gradient_ascent_final.pth train_gradient_ascent.py --ascent_epochs 1 --finetune_epochs 5
run salun       salun_final.pth        train_salun.py --epochs 10 --sparsity 0.5 --lambda_random 0.5
run ssd         ssd_unlearned.pth      unlearn_ssd.py --alpha 0.7 --lambda_thr 10.0
run bad_teacher bad_teacher_final.pth  train_bad_teacher.py --epochs 20 --lambda_ce 0.5
run lotus       lotus_final.pth        train_lotus.py --epochs 5 --lr 1e-5 --alpha 2.0 --subset_size 0.3
run munba       munba_final.pth        train_munba.py --epochs 10 --lr 0.005 --momentum 0.9 --weight_decay 0.0005
run loft        loft_U.pt              train_loft.py --steps 100 --lr 1.0 --weight_decay 0.05 --energy 0.95
run lora        lora_unlearn_final.pth train_lora_unlearn.py --epochs 20 --rank 4 --alpha 8
if [ -f logs/${P}_lora/lora_unlearn_final.pth ] && [ ! -f logs/${P}_lora/lora_merged.pth ]; then
  python -u scripts/rebuttal/merge_and_eval_lora.py --in_ckpt logs/${P}_lora/lora_unlearn_final.pth \
    --out_ckpt logs/${P}_lora/lora_merged.pth > "$LOGDIR/${P}_lora_merge.log" 2>&1 \
    && echo "[ok] lora merge" || echo "[FAIL] lora merge"
fi

run scrub_00001 scrub_final.pth        train_scrub.py --rounds 5 --max_epochs 2 --min_epochs 3 --alpha 1.0 --beta 1.0 --gamma 1.0 SOLVER.BASE_LR 0.0001
run neggrad_00005 neggrad_plus_final.pth    train_neggrad_plus.py --epochs 20 --alpha 1.0 --beta 0.5 SOLVER.BASE_LR 0.0005
echo "${P}_BASELINES_DONE $(date +%H:%M)"
