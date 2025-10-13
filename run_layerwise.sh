#!/bin/bash
# filepath: /home/davidhuang/MUL/run_layerwise_full.sh

echo "=========================================="
echo "LayerWise 完整遺忘實驗 (10-50類別)"
echo "時間: $(date)"
echo "=========================================="

# 設定基本參數
BASE_MODEL_PATH="/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth"
OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
SEED=42

# 創建必要目錄
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "實驗配置:"
echo "- 基礎模型: $BASE_MODEL_PATH"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "- 日誌目錄: $LOG_DIR"
echo "- 隨機種子: $SEED"
echo "- 遺忘類別數量: 10, 20, 30, 40, 50"
echo "- MIA評估: 啟用"
echo "- 增強評估: 啟用"
echo ""

# 顯示預計遺忘的類別
echo "預計遺忘的類別組合："
for N in 10 20 30 40 50; do
    CLASSES=$(python3 -c "
import random
random.seed($SEED + $N)
print(','.join(map(str, sorted(random.sample(range(100), $N)))))
")
    echo "- ${N}類: ${CLASSES}"
done
echo ""

# 檢查模型文件是否存在
if [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "❌ 錯誤: 找不到模型文件 $BASE_MODEL_PATH"
    echo "請檢查模型路徑是否正確"
    exit 1
fi

# 檢查CUDA是否可用
echo "檢查CUDA環境..."
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU數量: {torch.cuda.device_count()}'); print(f'當前設備: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}');"

echo ""
echo "開始執行LayerWise完整實驗..."
echo "預估總時間: 2-4小時 (取決於硬件配置)"
echo "=========================================="

# 執行LayerWise實驗
python main/main_layerwise.py \
    --model_path "$BASE_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --forget_class_counts 10 20 30 40 50 \
    --seed $SEED \
    --strategies all \
    --unlearn_epochs 30 \
    --unlearn_lr 5e-5 \
    --weight_decay 0.01 \
    --batch_size 128 \
    --num_workers 4 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_lr_scheduler onecycle \
    --gs_weight_decay 0.05 \
    --run_mia \
    --mia_epochs 30 \
    --mia_lr 1e-3 \
    --mia_use_scheduler \
    --run_enhanced_eval \
    --save_detailed_logs \
    --verbose \
    2>&1 | tee "$LOG_DIR/layerwise_full_experiment.log"

# 檢查執行結果
RETURN_CODE=$?

echo ""
echo "=========================================="

if [ $RETURN_CODE -eq 0 ]; then
    echo "✅ LayerWise 完整實驗成功完成！"
    echo "----------------------------------------"
    
    # 查找最新的實驗目錄
    LATEST_EXP_DIR=$(find $OUTPUT_DIR -name "layerwise_*" -type d | sort | tail -1)
    
    if [ -n "$LATEST_EXP_DIR" ]; then
        echo "📁 實驗結果目錄: $LATEST_EXP_DIR"
        echo ""
        
        # 顯示結果文件結構
        echo "📊 生成的主要文件:"
        echo "├── 實驗配置:"
        find "$LATEST_EXP_DIR" -name "experiment_config.json" -o -name "*_args.txt" | sed 's/^/│   ├── /'
        echo "├── 實驗結果:"
        find "$LATEST_EXP_DIR" -name "*.json" -path "*/results/*" | head -5 | sed 's/^/│   ├── /'
        [ $(find "$LATEST_EXP_DIR" -name "*.json" -path "*/results/*" | wc -l) -gt 5 ] && echo "│   └── ... (更多結果文件)"
        echo "├── 總結報告:"
        find "$LATEST_EXP_DIR" -name "*.csv" -o -name "*report.txt" | sed 's/^/│   ├── /'
        echo "└── 實驗日誌:"
        find "$LATEST_EXP_DIR" -name "*.log" | sed 's/^/    └── /'
        
        echo ""
        
        # 顯示總結CSV的簡要信息
        SUMMARY_CSV=$(find "$LATEST_EXP_DIR" -name "*summary.csv" | head -1)
        if [ -f "$SUMMARY_CSV" ]; then
            echo "📈 實驗結果總結:"
            echo "總共執行的實驗數量:"
            tail -n +2 "$SUMMARY_CSV" | wc -l | xargs echo "  ├── 實驗數量:"
            echo "  ├── 策略類型:"
            tail -n +2 "$SUMMARY_CSV" | cut -d',' -f2 | sort | uniq | wc -l | xargs echo "    └── 策略數:"
            echo "  └── 遺忘類別配置:"
            tail -n +2 "$SUMMARY_CSV" | cut -d',' -f1 | sort | uniq | xargs echo "    └──"
            echo ""
            
            echo "🏆 最佳ZRF分數 (前3名):"
            echo "策略,遺忘類別,ZRF分數"
            tail -n +2 "$SUMMARY_CSV" | sort -t',' -k9 -nr | head -3 | cut -d',' -f1,2,9 | nl -w2 -s'. '
        fi
        
        echo ""
        echo "🔍 詳細查看方式:"
        echo "  ├── 查看CSV總結: cat $SUMMARY_CSV"
        echo "  ├── 查看詳細報告: cat $LATEST_EXP_DIR/results/*report.txt"
        echo "  ├── 查看實驗日誌: cat $LATEST_EXP_DIR/logs/*.log"
        echo "  └── 查看配置文件: cat $LATEST_EXP_DIR/experiment_config.json"
        
    else
        echo "⚠️  找不到實驗結果目錄，請檢查 $OUTPUT_DIR"
    fi
    
else
    echo "❌ LayerWise 實驗執行失敗 (返回碼: $RETURN_CODE)"
    echo "----------------------------------------"
    echo "🔧 錯誤排查建議:"
    echo "1. 檢查完整錯誤日誌: tail -50 $LOG_DIR/layerwise_full_experiment.log"
    echo "2. 檢查模型路徑是否正確: ls -la $BASE_MODEL_PATH"
    echo "3. 檢查CUDA內存是否足夠: nvidia-smi"
    echo "4. 檢查Python依賴是否完整: pip list | grep torch"
    echo ""
    echo "📋 常見錯誤解決方案:"
    echo "  ├── CUDA OOM: 減少batch_size或使用更少的worker"
    echo "  ├── 模型加載失敗: 檢查模型文件是否損壞"
    echo "  ├── 導入錯誤: 檢查Python路徑和依賴安裝"
    echo "  └── 權限錯誤: 檢查輸出目錄寫入權限"
fi

echo ""
echo "📝 完整執行日誌: $LOG_DIR/layerwise_full_experiment.log"
echo "⏰ 實驗結束時間: $(date)"
echo "=========================================="