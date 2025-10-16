import os
import sys
import optuna
import logging
import argparse
import pandas as pd
import torch.multiprocessing as mp
from datetime import datetime
from main.pretrain_gs_model import run_training_trial

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def objective(trial):
    lr = trial.suggest_float("gs_lr", 5e-5, 1e-3, log=True)
    wd = trial.suggest_float("gs_weight_decay", 0.05, 0.2)
    alpha = trial.suggest_categorical("mix_alpha", [0.1, 0.2, 0.3])
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.05, 0.1, 0.15]) 
    
    output_dir = f"./checkpoints_pretrain/gs_tuning/trial_{trial.number}"
    os.makedirs(output_dir, exist_ok=True)

    args = argparse.Namespace(
        forget_count = 50,
        output_dir = f"./checkpoints_pretrain/gs_tuning/trial_{trial.number}",
        gs_epochs = 200,
        gs_lr = lr,
        gs_weight_decay = wd,
        mix_alpha = alpha,
        label_smoothing = label_smoothing,
        batch_size = 512,
        gs_scheduler = "onecycle",
        use_mixup = True,
        gs_use_ema = True,
        gs_ema_decay = 0.999,
        device = "cuda",
        gs_min_lr = 5e-5
    )
    
    print(f"\n===== [Trial #{trial.number}] STARTING =====")
    print(f"  Params: lr={args.gs_lr:.5f}, wd={args.gs_weight_decay:.5f}, alpha={args.mix_alpha}, ls={args.label_smoothing}")
    
    log_path = os.path.join(output_dir, "training.log")
    print(f"  Log file will be saved to: {log_path}")
    
    original_stdout = sys.stdout # 保存原始的標準輸出

    try:
        
        with open(log_path, 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file
            
            final_acc = run_training_trial(args)
        
        sys.stdout = original_stdout
        
        print(f"  [Trial #{trial.number}] SUCCESS! Final Accuracy = {final_acc:.4f}")
        return final_acc

    except Exception as e:
        sys.stdout = original_stdout
        print(f"  [Trial #{trial.number}] CRASHED! An exception occurred.")
        
        # 打印詳細的錯誤追蹤，方便除錯
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 建立一個 "study" 物件，並指定一個 Pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    study.optimize(objective, n_trials=20)
    
    # ... (後續儲存 TXT 和 CSV 的程式碼保持不變) ...
    print("\n\n===== OPTIMIZATION FINISHED, SAVING RESULTS... =====")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"./checkpoints_pretrain/gs_tuning_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    best_trial = study.best_trial
    summary_path = os.path.join(results_dir, "tuning_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Hyperparameter Tuning Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Best Trial Number: {best_trial.number}\n")
        f.write(f"Best Value (Max Accuracy): {best_trial.value:.4f}\n\n")
        f.write("Best Parameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  --{key}: {value}\n")
    print(f"✅ Best parameters saved to: {summary_path}")

    trials_df = study.trials_dataframe()
    csv_path = os.path.join(results_dir, "tuning_trials.csv")
    trials_df.to_csv(csv_path, index=False)
    print(f"✅ Full trial data saved to: {csv_path}")

    print("\nBest trial:")
    print(f"  Value (Max Accuracy): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")