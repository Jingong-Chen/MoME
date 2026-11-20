#!/usr/bin/env bash
# Portable form of the script that produced the paper's numbers. It assumes the repository root as
# the working directory and MOME_DATA pointing at the datasets; everything else is as it was run.
# The reference band under the disjoint families, on all three benchmarks.
#
# A target row is scored against four references, so each member of the family has to be scored
# against four others for the band to mean the same thing. That needs five members per dataset.
set -o pipefail
: "${MOME_DATA:=./data}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD"
LOGDIR=${LOGDIR:-logs/_repro}
mkdir -p "$LOGDIR"
SUB=(MOME.FORGET_MODE "'sub_id'" MOME.FORGET_ID_FRAC 0.5 MOME.SPLIT_SEED 42 MOME.DR_K 4)
MCFG=configs/Market/vit_transreid_stride_mome_v3.yml
RCFG=configs/RegDB/vit_transreid_stride_cross_mome_v3.yml
VCFG=configs/VeRi/vit_transreid_stride_mome_v3.yml

train_market () { # seed
  [ -f logs/xref_market_s$1/transformer_120.pth ] && return 0
  python -u train_retrain_baseline.py --config_file $MCFG \
    MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
    MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 MOME.EVAL_TRAIN_MAP False \
    "${SUB[@]}" SOLVER.SEED $1 OUTPUT_DIR logs/xref_market_s$1 > "$LOGDIR/xref_market_s$1.log" 2>&1
}
train_regdb () {
  [ -f logs/xref_regdb_s$1/transformer_120.pth ] && return 0
  python -u scripts/run_with_regdb_cross.py --script train_retrain_baseline.py --trial 1 --test_mode v2t -- \
    --config_file $RCFG MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
    MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth MODEL.SIE_CAMERA False \
    SOLVER.OPTIMIZER_NAME "'AdamW'" SOLVER.BASE_LR 0.0003 SOLVER.WARMUP_EPOCHS 3 \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 120 MOME.EVAL_TRAIN_MAP False \
    "${SUB[@]}" MOME.DR_K_STRATIFY_CAM True SOLVER.SEED $1 OUTPUT_DIR logs/xref_regdb_s$1 \
    > "$LOGDIR/xref_regdb_s$1.log" 2>&1
}
train_veri () {
  [ -f logs/xref_veri_s$1/transformer_120.pth ] && return 0
  python -u train_retrain_baseline.py --config_file $VCFG \
    MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
    MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 MOME.EVAL_TRAIN_MAP False \
    "${SUB[@]}" SOLVER.SEED $1 OUTPUT_DIR logs/xref_veri_s$1 > "$LOGDIR/xref_veri_s$1.log" 2>&1
}

band () { # dataset cfg outdir  member_paths...
  local ds=$1 cfg=$2 out=$3; shift 3
  local members=("$@")
  mkdir -p "$out"
  for i in "${!members[@]}"; do
    local t=${members[$i]}
    local tag=$(basename "$(dirname "$t")")
    local aj="$out/advmia_band_$tag.json"
    [ -f "$aj" ] && { echo "[have] $ds band $tag"; continue; }
    local refs=""
    for j in "${!members[@]}"; do
      [ "$j" = "$i" ] && continue
      refs="$refs,${members[$j]}"
    done
    refs=${refs#,}
    local modes=standard,standard,standard,standard
    if [ "$ds" = regdb ]; then
      python -u scripts/run_with_regdb_cross.py --script eval_advanced_mia.py --trial 1 --test_mode v2t -- \
        --config_file "$cfg" --mode standard --ref_ckpts "$refs" --ref_modes "$modes" --out_json "$aj" \
        --dump_scores "$out/scores_band_$tag.npz" \
        MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
        MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth MODEL.SIE_CAMERA False \
        "${SUB[@]}" MOME.DR_K_STRATIFY_CAM True TEST.WEIGHT "$t" OUTPUT_DIR "$out" \
        > "$out/adv_band_$tag.log" 2>&1
    else
      python -u eval_advanced_mia.py --config_file "$cfg" --mode standard \
        --ref_ckpts "$refs" --ref_modes "$modes" --out_json "$aj" \
        --dump_scores "$out/scores_band_$tag.npz" \
        MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR "$MOME_DATA" \
        MODEL.PRETRAIN_PATH ./jx_vit_base_p16_224-80ecf9dd.pth \
        "${SUB[@]}" TEST.WEIGHT "$t" OUTPUT_DIR "$out" > "$out/adv_band_$tag.log" 2>&1
    fi
    python3 -c "
import json
a=json.load(open('$aj'))
print('[band] $ds %-22s ent %.4f  RMIA %.4f  U-LiRA %.4f  TPR@1%% %.4f' % (
 '$tag', a['entropy']['auc'], a['rmia_lf']['auc'], a['u_lira_lf']['auc'], a['rmia_lf']['tpr_at_0.01']))"
  done
  echo "${ds^^}_BAND_DONE"
}

train_market 2005 || { echo MARKET_2005_FAILED; exit 1; }
band market $MCFG logs/xref_market \
  logs/xref_market_s2001/transformer_120.pth logs/xref_market_s2002/transformer_120.pth \
  logs/xref_market_s2003/transformer_120.pth logs/xref_market_s2004/transformer_120.pth \
  logs/xref_market_s2005/transformer_120.pth

train_regdb 2005 || { echo REGDB_2005_FAILED; exit 1; }
band regdb $RCFG logs/xref_regdb \
  logs/xref_regdb_s2001/transformer_120.pth logs/xref_regdb_s2002/transformer_120.pth \
  logs/xref_regdb_s2003/transformer_120.pth logs/xref_regdb_s2004/transformer_120.pth \
  logs/xref_regdb_s2005/transformer_120.pth

train_veri 2003 || { echo VERI_2003_FAILED; exit 1; }
band veri $VCFG logs/xref_veri \
  logs/kmix_veri_retrain_s789/transformer_120.pth logs/kmix_veri_retrain_s1024/transformer_120.pth \
  logs/xref_veri_s2001/transformer_120.pth logs/xref_veri_s2002/transformer_120.pth \
  logs/xref_veri_s2003/transformer_120.pth
echo XREF_BAND_ALL_DONE
