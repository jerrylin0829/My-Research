#!/usr/bin/env bash
###############################################################################
#  File   : run_weightpruning_10-50.sh
#  Usage  : bash run_weightpruning_10-50.sh
#  Note   : 串行執行 WeightPruning：10、20、30、40、50 類別遺忘實驗
###############################################################################
set -euo pipefail

# --- 1. 定義實驗的關鍵元信息 ---
METHOD="WeightPruning"
FORGET_COUNTS="10 20"
STRATEGY="gradient"
PRUNE_RATIOS="0.7 0.9 0.95"
TARGET_LAYERS="head,late_blocks"

# --- 2. 組合出描述性的實驗名稱 ---
NOW=$(date +%Y%m%d-%H%M%S)
EXP_NAME="${METHOD}_Forget[${FORGET_COUNTS}]_Strategy[${STRATEGY}]_Layers[${TARGET_LAYERS}]_${NOW}"

# --- 3. 設定根目錄和輸出目錄 ---
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/checkpoints/${EXP_NAME}" # 使用新的實驗名稱
BEST_MODEL="/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth"  
PYTHON_EXE="python"
SEED=42

mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/logs"

# --- 4. 執行實驗 (迴圈不變) ---
for N in ${FORGET_COUNTS}; do
  echo "▶ 開始 WeightPruning 遺忘 ${N} 個類別"

  ${PYTHON_EXE} "${ROOT_DIR}/main/main_weightpruning.py" \
    --model_path "${BEST_MODEL}" \
    --output_dir "${OUTPUT_ROOT}" \
    --forget_class_counts ${N} \
    --strategies ${STRATEGY} \
    --prune_ratios ${PRUNE_RATIOS} \
    --prune_target_layers "${TARGET_LAYERS}" \
    --batch_size 512 \
    --unlearn_epochs 20 \
    --gs_epochs 300 \
    --gs_lr 4e-4 \
    --gs_min_lr 5e-5 \
    --gs_scheduler onecycle \
    --gs_use_ema \
    --gs_ema_decay 0.999 \
    --gs_weight_decay 0.05 \
    --run_enhanced_eval \
    --no_mia \
    --use_mixup
    2>&1 | tee "${OUTPUT_ROOT}/forget${N}_run.log"

  echo "✅ 完成 WeightPruning N=${N}，結果已寫入 ${OUTPUT_ROOT}/forget${N}"

done

echo "🎉 WeightPruning 串行實驗 10~50 類別全部完成"