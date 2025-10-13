#!/usr/bin/env bash
###############################################################################
#  File   : run_attention_10-50.sh
#  Usage  : bash run_attention_10-50.sh
#  Note   : 串行執行 AttentionTargeted：10、20、30、40、50 類別遺忘實驗
###############################################################################
set -euo pipefail

NOW=$(date +%Y%m%d-%H%M%S)
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/checkpoints/attention_${NOW}"
BEST_MODEL="/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth"  
PYTHON_EXE="python"
SEED=42

mkdir -p "${OUTPUT_ROOT}" "${ROOT_DIR}/logs"

for N in 10 20 30 40 50; do
  echo "====================================================================="
  echo "▶ 開始 AttentionTargeted 遺忘 ${N} 個類別"

  # 1) 產生遺忘類別清單（固定 SEED 以利重現）
  CLASSES=$(
    ${PYTHON_EXE} - <<PY
import random; random.seed(${SEED}+${N})
print(",".join(map(str, sorted(random.sample(range(100), ${N})))))
PY
  )
  echo "   類別清單：${CLASSES}"

  # 2) 建立輸出資料夾
  RUN_DIR="${OUTPUT_ROOT}/forget${N}"
  mkdir -p "${RUN_DIR}"

  # 3) 執行主程式
  ${PYTHON_EXE} "${ROOT_DIR}/main/main_attention.py" \
    --model_path "${BEST_MODEL}" \
    --output_dir "${RUN_DIR}" \
    --forget_classes "${CLASSES}" \
    --strategies all \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_scheduler onecycle \
    --gs_weight_decay 0.05 \
    --mia_epochs 50 \
    --mia_lr 1e-3 \
    --log_file "${ROOT_DIR}/logs/attention_${N}_${NOW}.log" \
    2>&1 | tee "${RUN_DIR}/run.log"

  echo "✅ 完成 AttentionTargeted N=${N}，結果已寫入 ${RUN_DIR}"
  echo

done

echo "🎉 AttentionTargeted 串行實驗 10~50 類別全部完成"
