#!/usr/bin/env bash
###############################################################################
#  File   : run_all_strategies.sh
#  Usage  : bash run_all_strategies.sh
#  Note   : 依序執行 magnitude_reset 和 gradient 策略的批量實驗
###############################################################################
set -euo pipefail

echo "=========================================================="
echo "🚀 開始執行所有 WeightPruning 策略批量實驗"
echo "=========================================================="
echo

# --- 實驗 1: Magnitude Reset ---
echo "▶️  (1/2) 即將執行 Magnitude Reset 策略..."
echo "----------------------------------------------------------"
bash ./run_weightpruning_10-50_magnitude.sh
echo "✅  (1/2) Magnitude Reset 策略執行完畢。"
echo "----------------------------------------------------------"
echo

# --- 實驗 2: Gradient ---
echo "▶️  (2/2) 即將執行 Gradient 策略..."
echo "----------------------------------------------------------"
bash ./run_weightpruning_10-50_gradient.sh
echo "✅  (2/2) Gradient 策略執行完畢。"
echo "----------------------------------------------------------"
echo

echo "🎉🎉🎉 所有策略均已執行完成！ 🎉🎉🎉"