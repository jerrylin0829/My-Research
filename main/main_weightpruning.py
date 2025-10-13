"""
WeightPruning 遺忘實驗主程式
"""
import os
import sys
import time
import json
import argparse
import logging
import torch
import torch.nn as nn
import shutil
import numpy as np
from datetime import datetime
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_LSA_classattn import VisionTransformer
from methods.weight_pruning_unlearner import WeightPruningUnlearner
from mul_vit_1 import MembershipInferenceAttack, MachineUnlearning
from utils.evaluation_utils import (
    enhanced_unlearning_evaluation, 
    print_enhanced_evaluation_report,
    calculate_model_size_difference,
)
from utils.storage_utils import ExperimentStorage
from mia_visualization import visualize_mia_features

def setup_logging(log_file):
    """設置日誌系統"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)


class WeightPruningExperiment:
    """WeightPruning 遺忘實驗管理器 - 使用統一的Storage系統"""
    
    def __init__(self, args):
        self.args = args
        self.base_model_path = args.model_path
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 使用統一的儲存系統
        self.storage = ExperimentStorage(args.output_dir)
        
        # 創建完整實驗結構
        self.exp_dir, self.exp_name = self.storage.create_complete_experiment_structure("weightpruning")
        
        # 設置日誌系統
        self.logger, self.log_file = self.storage.setup_logging_system(self.exp_dir, self.exp_name)
        
        # 設置TensorBoard
        self.tensorboard_dir, self.main_writer = self.storage.setup_tensorboard(self.exp_dir)
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.fine_tune_epochs = args.unlearn_epochs
        
        # 黃金標準參數
        self.gs_epochs = args.gs_epochs
        self.gs_lr = args.gs_lr
        self.gs_scheduler = args.gs_scheduler
        self.gs_min_lr = args.gs_min_lr
        self.gs_weight_decay = args.gs_weight_decay
        self.use_mixup = args.use_mixup
        self.use_cutmix = args.use_cutmix
        self.mix_alpha = args.mix_alpha
        self.use_logit_penalty = args.use_logit_penalty
        self.use_forget_uniform = args.use_forget_uniform
        self.use_forget_distill = args.use_forget_distill

        self.prune_ratios = args.prune_ratios

        def _parse_target_layers(s):
            return [t.strip() for t in s.split(',') if t.strip()]
        
        self.prune_target_layers = _parse_target_layers(args.prune_target_layers)

        # 評估參數
        # self.run_mia = args.run_mia
        # self.mia_epochs = args.mia_epochs
        self.run_mia = False  # 🔧 暫時禁用 MIA
        self.mia_epochs = 0
        self.run_enhanced_eval = args.run_enhanced_eval
        
        # MIA 額外參數（如果腳本需要）
        # self.mia_lr = getattr(args, 'mia_lr', 1e-3)
        # self.mia_use_scheduler = getattr(args, 'mia_use_scheduler', True)
        self.mia_lr = 1e-3
        self.mia_use_scheduler = True
        
        # 模型參數
        self.model_args = {
            'img_size': 32,
            'patch_size': 4,
            'in_chans': 3,
            'num_classes': 100,
            'embed_dim': 300,
            'depth': 10,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.05,
            'is_LSA': True
        }
        
        # 保存完整配置並記錄實驗開始
        config_file, self.config = self.storage.save_complete_experiment_config(
            self.exp_dir, "WeightPruning", args, self.model_args
        )
        
        # 統一的實驗開始日誌
        self.storage.log_experiment_start(
            self.logger, "WeightPruning", self.exp_name, self.exp_dir, self.config, args
        )
        
        # 新增 GS (Retrain model) 的快取，避免重複訓練
        self.gs_cache = {}

        self.logger.info(f"TensorBoard: tensorboard --logdir={self.tensorboard_dir}")
        print(f"🚀 WeightPruning實驗開始 - 目錄: {self.exp_dir}")

    def _prepare_gold_standard(self, forget_classes, forget_key):
        """專門訓練、評估並保存一個黃金標準模型"""
        self.logger.info(f"===== 開始準備 {forget_key} 的黃金標準模型 =====")
        
        # 建立一個暫時的 MachineUnlearning 實例來執行訓練
        temp_output_dir = f"./temp_gs_training_{forget_key}"
        mul = MachineUnlearning(
            model_class=VisionTransformer,
            model_args=self.model_args,
            device=self.device,
            output_dir=temp_output_dir,
            args=self.args
        )
        
        mul.prepare_data(dataset_name='CIFAR100', forget_classes=forget_classes)
        
        # 訓練 GS 模型
        mul.train_gold_standard(
            epochs=self.gs_epochs,
            lr=self.gs_lr,
            lr_scheduler_type=self.gs_scheduler,
            min_lr=self.gs_min_lr,
            weight_decay=self.gs_weight_decay,
            use_mixup=self.use_mixup,
            use_cutmix=self.use_cutmix,
            mix_alpha=self.mix_alpha,
            use_logit_penalty=self.use_logit_penalty
        )
        
        # 評估 GS 模型
        gs_retain_acc = mul._evaluate_retain(mul.retrained_model)
        gs_forget_acc = mul._evaluate_forget(mul.retrained_model)
        
        # 定義儲存路徑並保存模型
        gs_model_path = os.path.join(self.exp_dir, "models", f"gs_model_{forget_key}.pth")
        torch.save(mul.retrained_model.state_dict(), gs_model_path)
        self.logger.info(f"黃金標準模型已保存至: {gs_model_path}")

        # 清理暫存目錄
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        # 返回包含路徑和指標的字典
        return {
            "model_path": gs_model_path,
            "retrained_model_obj": mul.retrained_model, # 返回模型物件以供後續使用
            "metrics": {
                "gs_retain_acc": gs_retain_acc,
                "gs_forget_acc": gs_forget_acc
            },
            "retraining_time": mul.results.get('retraining_time', 0)
        }
    
    def run_all_experiments(self, forget_class_counts, selected_strategies=None):
        """批量執行所有實驗組合（高效版）"""
        # === 階段一：預先計算並快取所有黃金標準模型 ===
        self.logger.info("="*80)
        self.logger.info("階段一：開始準備所有黃金標準模型...")
        self.logger.info("="*80)
        
        import random
        for forget_count in forget_class_counts:
            forget_key = f"forget{forget_count}"
            if forget_key not in self.gs_cache:
                # 為了可重現性，固定隨機種子來生成遺忘類別
                random.seed(42 + forget_count)
                forget_classes = set(random.sample(range(100), forget_count))
                
                gs_info = self._prepare_gold_standard(forget_classes, forget_key)
                self.gs_cache[forget_key] = gs_info
                # 我們需要 forget_classes 本身以供後續使用
                self.gs_cache[forget_key]['forget_classes'] = forget_classes
        
        self.logger.info("✅ 所有黃金標準模型準備完畢！")

        # === 階段二：執行所有遺忘策略 ===
        self.logger.info("="*80)
        self.logger.info("階段二：開始執行所有遺忘策略...")
        self.logger.info("="*80)

        # 獲取策略配置
        temp_unlearner = WeightPruningUnlearner(None, self.device)
        all_strategies = temp_unlearner.get_strategy_configs(prune_ratios=self.prune_ratios, 
                                                             target_layers=self.prune_target_layers)
        
        # 篩選策略 (邏輯不變)
        if selected_strategies is None or 'all' in selected_strategies:
            strategies = all_strategies
        else:
            strategies = {name: config for name, config in all_strategies.items()
                         if any(strat in name for strat in selected_strategies)}
        
        all_results = {}
        total_experiments = len(forget_class_counts) * len(strategies)
        experiment_step = 0

        with tqdm(total=total_experiments, desc="批量實驗進度") as pbar:
            for forget_count in forget_class_counts:
                forget_key = f"forget{forget_count}"
                all_results[forget_key] = {}
                
                # 從快取中獲取 GS 資訊和遺忘類別
                gs_info = self.gs_cache[forget_key]
                forget_classes = gs_info['forget_classes']
                
                for strategy_name, strategy_config in strategies.items():
                    experiment_step += 1
                    try:
                        pbar.set_description(f"執行 {forget_key}_{strategy_name}")
                        
                        # 執行單一遺忘實驗，傳入已準備好的 GS 資訊
                        result = self.run_experiment(forget_classes, strategy_name, strategy_config, gs_info, global_step=experiment_step)
                        
                        all_results[forget_key][strategy_name] = result
                        
                        self._write_tensorboard_summary(forget_key, strategy_name, result, experiment_step)
                        
                    except Exception as e:
                        error_msg = str(e)
                        all_results[forget_key][strategy_name] = {'error': error_msg}
                        self.logger.error(f"❌ 失敗: {forget_key} - {strategy_name}: {error_msg}", exc_info=True)
                    
                    pbar.update(1)
        
        # 產生報告
        self._generate_comprehensive_csv(all_results)
        self._generate_detailed_report(all_results)
        self._perform_statistical_analysis(all_results)
        
        self.main_writer.close()
        self.logger.info("🎉 WeightPruning實驗全部完成！")
        print(f"📁 結果目錄: {self.exp_dir}")
        return self.exp_dir, all_results

    def _write_tensorboard_summary(self, forget_key, strategy_name, result, step):
        """寫入 TensorBoard 摘要"""
        if 'error' in result:
            return
            
        prefix = f"{forget_key}_{strategy_name}"
        
        # 基礎指標
        perf = result.get('performance_metrics', {})
        self.main_writer.add_scalar(f"{prefix}/Original_Retain_Acc", perf.get('original_retain_acc', 0), step)
        self.main_writer.add_scalar(f"{prefix}/Unlearned_Retain_Acc", perf.get('unlearned_retain_acc', 0), step)
        self.main_writer.add_scalar(f"{prefix}/GS_Retain_Acc", perf.get('gs_retain_acc', 0), step)
        
        # 遺忘指標
        unlearn = result.get('unlearning_metrics', {})
        self.main_writer.add_scalar(f"{prefix}/ZRF_Score", unlearn.get('zrf_score', 0), step)
        self.main_writer.add_scalar(f"{prefix}/Forget_Effectiveness", unlearn.get('forget_effectiveness', 0), step)
        self.main_writer.add_scalar(f"{prefix}/Retain_Preservation", unlearn.get('retain_preservation', 0), step)
        
        # 時間
        timing = result.get('timing', {})
        self.main_writer.add_scalar(f"{prefix}/Unlearning_Time", timing.get('unlearning_time', 0), step)
        
        self.main_writer.flush()

    def _generate_comprehensive_csv(self, all_results):
        """生成詳細 CSV 比較表"""
        rows = []
        
        for forget_key, strategies_results in all_results.items():
            for strategy_name, result in strategies_results.items():
                if 'error' in result:
                    continue
                    
                perf = result.get('performance_metrics', {})
                unlearn = result.get('unlearning_metrics', {})
                timing = result.get('timing', {})
                model_analysis = result.get('model_analysis', {})
                enhanced = result.get('enhanced_metrics', {})
                mia = result.get('mia_results', {})
                
                row = {
                    'Forget_Classes': forget_key,
                    'Strategy': strategy_name,
                    'Original_Retain_Acc': perf.get('original_retain_acc', 0),
                    'Original_Forget_Acc': perf.get('original_forget_acc', 0),
                    'Unlearned_Retain_Acc': perf.get('unlearned_retain_acc', 0),
                    'Unlearned_Forget_Acc': perf.get('unlearned_forget_acc', 0),
                    'GS_Retain_Acc': perf.get('gs_retain_acc', 0),
                    'GS_Forget_Acc': perf.get('gs_forget_acc', 0),
                    'ZRF_Score': unlearn.get('zrf_score', 0),
                    'KL_Divergence': unlearn.get('kl_divergence', 0),
                    'Forget_Effectiveness': unlearn.get('forget_effectiveness', 0),
                    'Retain_Preservation': unlearn.get('retain_preservation', 0),
                    'Unlearning_Time': timing.get('unlearning_time', 0),
                    'Retraining_Time': timing.get('retraining_time', 0),
                    'Sparsity_Increase': model_analysis.get('sparsity_increase', 0),
                    'Compression_Ratio': model_analysis.get('compression_ratio', 1),
                    'Cosine_Similarity': enhanced.get('cosine_similarity', 0),
                    # 'MIA_Accuracy': mia.get('accuracy', 0.5)  # 🔧 MIA 暫時禁用
                    'MIA_Accuracy': 0.5  # 預設值
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = f"{self.exp_dir}/results/comprehensive_comparison.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"CSV 比較表已保存: {csv_path}")
            return csv_path

    def _generate_detailed_report(self, all_results):
        """生成詳細文字報告"""
        report_path = f"{self.exp_dir}/results/detailed_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("WeightPruning 遺忘實驗詳細報告\n")
            f.write("="*50 + "\n\n")
            f.write(f"實驗時間: {datetime.now()}\n")
            f.write(f"實驗目錄: {self.exp_dir}\n\n")
            
            # 實驗概覽
            total_experiments = sum(len(strategies) for strategies in all_results.values())
            successful = sum(1 for strategies in all_results.values() 
                           for result in strategies.values() if 'error' not in result)
            
            f.write(f"實驗總數: {total_experiments}\n")
            f.write(f"成功實驗: {successful}\n")
            f.write(f"成功率: {successful/total_experiments*100:.1f}%\n\n")
            
            # 各實驗結果
            for forget_key, strategies_results in all_results.items():
                f.write(f"\n{forget_key.upper()} 結果:\n")
                f.write("-" * 30 + "\n")
                
                for strategy_name, result in strategies_results.items():
                    if 'error' in result:
                        f.write(f"  {strategy_name}: 失敗 - {result['error']}\n")
                        continue
                    
                    perf = result.get('performance_metrics', {})
                    unlearn = result.get('unlearning_metrics', {})
                    
                    f.write(f"  {strategy_name}:\n")
                    f.write(f"    保留準確率: {perf.get('original_retain_acc', 0):.2f}% → {perf.get('unlearned_retain_acc', 0):.2f}%\n")
                    f.write(f"    遺忘準確率: {perf.get('original_forget_acc', 0):.2f}% → {perf.get('unlearned_forget_acc', 0):.2f}%\n")
                    f.write(f"    黃金標準 (保留/遺忘): {perf.get('gs_retain_acc', 0):.2f}% / {perf.get('gs_forget_acc', 'N/A')}%\n")
                    f.write(f"    ZRF 分數: {unlearn.get('zrf_score', 0):.4f}\n")
                    f.write(f"    遺忘效果: {unlearn.get('forget_effectiveness', 0):.4f}\n")
                    f.write(f"    保留保持: {unlearn.get('retain_preservation', 0):.4f}\n\n")
        
        self.logger.info(f"詳細報告已保存: {report_path}")
        return report_path

    def _perform_statistical_analysis(self, all_results):
        """統計分析"""
        analysis_path = f"{self.exp_dir}/results/statistical_analysis.txt"
        
        # 收集有效結果
        valid_results = []
        for forget_key, strategies_results in all_results.items():
            for strategy_name, result in strategies_results.items():
                if 'error' not in result:
                    result_copy = result.copy()
                    result_copy['forget_key'] = forget_key
                    result_copy['strategy_name'] = strategy_name
                    valid_results.append(result_copy)
        
        if not valid_results:
            return
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("WeightPruning 統計分析\n")
            f.write("="*30 + "\n\n")
            
            # 最佳策略分析
            f.write("1. 最佳策略分析:\n")
            f.write("-" * 20 + "\n")
            
            # 按 ZRF 排序
            zrf_results = [(r['forget_key'], r['strategy_name'], 
                           r.get('unlearning_metrics', {}).get('zrf_score', 0)) 
                          for r in valid_results]
            zrf_results.sort(key=lambda x: x[2], reverse=True)
            
            f.write("按 ZRF 分數排序 (前5名):\n")
            for i, (forget_key, strategy, zrf) in enumerate(zrf_results[:5], 1):
                f.write(f"  {i}. {forget_key}_{strategy}: {zrf:.4f}\n")
            
            # 策略平均表現
            f.write("\n2. 策略平均表現:\n")
            f.write("-" * 20 + "\n")
            
            strategy_stats = {}
            for result in valid_results:
                strategy = result['strategy_name']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = []
                
                zrf = result.get('unlearning_metrics', {}).get('zrf_score', 0)
                strategy_stats[strategy].append(zrf)
            
            for strategy, zrf_scores in strategy_stats.items():
                if zrf_scores:
                    avg_zrf = np.mean(zrf_scores)
                    std_zrf = np.std(zrf_scores)
                    f.write(f"  {strategy}: {avg_zrf:.4f} ± {std_zrf:.4f} (n={len(zrf_scores)})\n")
            
            # 規模效應分析
            f.write("\n3. 規模效應分析:\n")
            f.write("-" * 20 + "\n")
            
            forget_stats = {}
            for result in valid_results:
                forget_key = result['forget_key']
                if forget_key not in forget_stats:
                    forget_stats[forget_key] = []
                
                zrf = result.get('unlearning_metrics', {}).get('zrf_score', 0)
                forget_stats[forget_key].append(zrf)
            
            for forget_key, zrf_scores in sorted(forget_stats.items()):
                if zrf_scores:
                    avg_zrf = np.mean(zrf_scores)
                    std_zrf = np.std(zrf_scores)
                    f.write(f"  {forget_key}: {avg_zrf:.4f} ± {std_zrf:.4f} (n={len(zrf_scores)})\n")
        
        self.logger.info(f"統計分析已保存: {analysis_path}")
    
    def run_experiment(self, forget_classes, strategy_name, strategy_config, gs_info, global_step=0):
        """運行單個WeightPruning實驗"""
        # 建立實驗專用 TensorBoard 目錄
        exp_id = f"forget{len(forget_classes)}_{strategy_name}"
        run_tb_dir = os.path.join(self.tensorboard_dir, exp_id)
        gold_tb_dir = os.path.join(run_tb_dir, "gold_standard")
        os.makedirs(run_tb_dir, exist_ok=True)
        os.makedirs(gold_tb_dir, exist_ok=True)
        
        # 建立 run summary writer
        writer = SummaryWriter(run_tb_dir)
        
        self.logger.info("="*80)
        self.logger.info(f"開始WeightPruning實驗: {strategy_name}")
        self.logger.info(f"遺忘類別: {sorted(forget_classes)}")
        self.logger.info(f"策略配置: {strategy_config}")
        self.logger.info("="*80)
        
        # 初始化MachineUnlearning框架
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_output_dir = f"./temp_weightpruning_{timestamp}"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            mul = MachineUnlearning(
                model_class=VisionTransformer,
                model_args=self.model_args,
                batch_size=self.batch_size,
                device=self.device,
                output_dir=temp_output_dir,
                log_dir=gold_tb_dir,
                args=self.args,
            )
            
            # 載入模型和準備數據
            mul.load_original_model(self.base_model_path)
            mul.prepare_data(dataset_name='CIFAR100', forget_classes=forget_classes)

            # 初始化隨機模型
            mul.init_random_model()
            
            # === 核心修改：直接從 gs_info 載入GS結果，不再訓練 ===
            self.logger.info("從快取載入黃金標準模型...")
            mul.retrained_model = gs_info["retrained_model_obj"]
            gs_metrics = gs_info["metrics"]
            gs_retain_acc = gs_metrics["gs_retain_acc"]
            gs_forget_acc = gs_metrics["gs_forget_acc"]
            retraining_time = gs_info["retraining_time"]
            # =======================================================

            # 評估原始模型
            original_retain_acc = mul._evaluate_retain(mul.original_model)
            original_forget_acc = mul._evaluate_forget(mul.original_model)

            dist_dir = os.path.join(run_tb_dir, "distributions")
            mul.plot_forget_distributions(mul.original_model,  "original",  dist_dir, exp_id=exp_id)
            mul.plot_forget_distributions(mul.retrained_model, "gold",      dist_dir, exp_id=exp_id)
        
            # 執行WeightPruning遺忘
            self.logger.info(f"執行WeightPruning遺忘: {strategy_name}")
            start_time = time.time()
            
            unlearner = WeightPruningUnlearner(mul.unlearned_model, self.device)
            mul.unlearned_model = unlearner.unlearn(
                retain_loader=mul.retain_train_loader,
                forget_loader=mul.forget_train_loader,
                fine_tune_epochs=self.fine_tune_epochs,
                **strategy_config,
            )

            unlearning_time = time.time() - start_time
            
            # 評估遺忘後的模型
            unlearned_retain_acc = mul._evaluate_retain(mul.unlearned_model)
            unlearned_forget_acc = mul._evaluate_forget(mul.unlearned_model)

            mul.plot_forget_distributions(mul.unlearned_model, "unlearned", dist_dir, exp_id=exp_id)
            if mul.random_model is not None:
                mul.plot_forget_distributions(mul.random_model,  "random",    dist_dir, exp_id=exp_id)
    
            # 增強評估
            enhanced_results = {}
            if self.run_enhanced_eval:
                self.logger.info("執行增強評估...")
                enhanced_results = enhanced_unlearning_evaluation(
                    mul.original_model, 
                    mul.unlearned_model, 
                    mul.retrained_model,
                    mul.random_model,
                    mul.retain_test_loader, 
                    mul.forget_test_loader, 
                    mul.retain_test_loader, 
                    self.device
                )
            
            # MIA評估
            mia_results = {'note': 'MIA disabled'}
            if self.run_mia:
                self.logger.info("執行MIA評估...")
                mia_results = self._run_mia(mul, plot_mode="eval")
            
            # 寫入基本摘要到 TensorBoard
            experiment_step = global_step

            writer.add_scalar("Original/Retain_Acc", original_retain_acc, experiment_step)
            writer.add_scalar("Original/Forget_Acc", original_forget_acc, experiment_step)
            writer.add_scalar("Unlearned/Retain_Acc", unlearned_retain_acc, experiment_step)
            writer.add_scalar("Unlearned/Forget_Acc", unlearned_forget_acc, experiment_step)
            writer.add_scalar("GoldStandard/Retain_Acc", gs_retain_acc, experiment_step)
            
            if enhanced_results:
                writer.add_scalar("Unlearning/ZRF", enhanced_results.get('zrf_score', 0.0), experiment_step)
                writer.add_scalar("Unlearning/Alignment", enhanced_results.get('alignment_score', 0.0), experiment_step)
                writer.add_scalar("Unlearning/KL", enhanced_results.get('kl_divergence', 0.0), experiment_step)

            writer.add_scalar("Timing/Unlearning_Time", unlearning_time, experiment_step)
            writer.add_scalar("Experiment/Retraining_Time", retraining_time, experiment_step)

            writer.flush()
            
            # === Debug Diagnostics (Unlearned) ===
            diag = {}
            try:
                print("\n=== Debug Diagnostics (Unlearned) ===")
                # 1) head 維度完整性（ confirm 仍為 100 類輸出 ）
                mul.check_head_integrity(mul.unlearned_model)

                # 2) 檢查 head.weight 是否有「整列全 0」的情況（zero-lock 痕跡）
                col_zero = mul.check_head_zero_columns(mul.unlearned_model, mul.forget_classes)
                if col_zero is not None:
                    total_zero_rows = int(col_zero.sum().item())
                    diag["head_zero_rows_total"] = total_zero_rows
                    if mul.forget_classes is not None:
                        fset = np.array(sorted(list(mul.forget_classes)))
                        # 安全：class id 可能未排序
                        forget_zero_rows = int(col_zero[fset].sum().item())
                        diag["head_zero_rows_in_forget"] = forget_zero_rows

                # 3) 預測在忘記測試集上的類別落點（落在 忘記/保留 的比例）
                r_forget, r_retain = mul.debug_forget_prediction_stats(mul.unlearned_model)
                diag["pred_ratio_forget"] = float(r_forget)
                diag["pred_ratio_retain"] = float(r_retain)

                # 4) 忘記集的「真實類 logit」直方圖（數據化 + 圖檔）
                diag_dir = os.path.join(run_tb_dir, "diagnostics")
                mul.plot_true_class_logit_hist(mul.unlearned_model, "unlearned", diag_dir)

                # 對照：original / gold 也各畫一張
                mul.plot_true_class_logit_hist(mul.original_model,  "original",  diag_dir)
                mul.plot_true_class_logit_hist(mul.retrained_model, "gold",      diag_dir)

                # 5) 忘記集上最常被預測的類別（Top-K 出現次數）
                mul.top_predicted_classes_on_forget(mul.unlearned_model, topk=10)

            except Exception as e:
                self.logger.warning(f"診斷程序失敗: {e}")

            # 整理結果
            results = {
                'experiment_info': {
                    'method': 'WeightPruning',
                    'strategy': strategy_name,
                    'forget_classes': sorted(forget_classes),
                    'num_forget_classes': len(forget_classes),
                    'timestamp': timestamp,
                    "use_mixup": self.use_mixup,
                    "use_cutmix": self.use_cutmix,
                    "mix_alpha": self.mix_alpha,
                    "use_logit_penalty": self.use_logit_penalty,
                },
                'strategy_config': strategy_config,
                'performance_metrics': {
                    'original_retain_acc': original_retain_acc,
                    'original_forget_acc': original_forget_acc,
                    'unlearned_retain_acc': unlearned_retain_acc,
                    'unlearned_forget_acc': unlearned_forget_acc,
                    'gs_retain_acc': gs_retain_acc,
                    'gs_forget_acc': gs_forget_acc
                },
                'enhanced_metrics': enhanced_results,
                'mia_results': mia_results,
                'timing': {
                    'unlearning_time': unlearning_time,
                    'retraining_time': retraining_time
                }
            }
            
            unlearned_model_path = f"{self.exp_dir}/results/forget{len(forget_classes)}_{strategy_name}_unlearned_model.pth"
            torch.save(mul.unlearned_model.state_dict(), unlearned_model_path)
            results['unlearned_model_path'] = unlearned_model_path

            return results
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"WeightPruning實驗失敗: {error_msg}", exc_info=True)
            return {'error': error_msg, 'strategy': strategy_name}
        
        finally:
            # 清理臨時目錄
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            writer.close()
    
    def _run_mia(self, mul, plot_mode="eval"):
        """運行完整的 MIA 評估 (mul_vit_1)"""
        try:
            mia = MembershipInferenceAttack(mul.unlearned_model, self.device)
            
            test_dataset = datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            )
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            # 準備資料：retain+test 用於訓練，retain+forget+test 用於評估
            X_train, y_train, X_test, y_test, original_labels_test = mia.prepare_attack_data(
                mul.retain_test_loader,
                mul.forget_test_loader,
                test_loader
            )

            # 從 X_train 裡再切 8:2 做驗證
            split = int(0.8 * len(X_train))
            X_tr, y_tr = X_train[:split], y_train[:split]
            X_val, y_val = X_train[split:], y_train[split:]

            # 訓練攻擊器 (以 AUC 驅動早停)
            mia.train_attack_model(
                X_tr, y_tr, X_val, y_val,
                epochs=self.mia_epochs,
                learning_rate=self.mia_lr,
                use_scheduler=self.mia_use_scheduler
            )

            # 最終評估 (retain+forget+test)
            mia_results = mia.evaluate_attack(X_test, y_test, original_labels_test)
            self.storage.save_results_to_json(
                self.exp_dir, mia_results, "mia_results.json"
            )
            
            # 視覺化
            self.logger.info("🎨 開始 MIA 特徵視覺化分析...")
            visualization_dir = os.path.join(self.exp_dir, "mia_visualization")
            os.makedirs(visualization_dir, exist_ok=True)

            # 重新提取特徵 (retain / forget / test)
            retain_features, retain_labels_orig = mia.extract_features(mul.unlearned_model, mul.retain_test_loader)
            forget_features, forget_labels_orig = mia.extract_features(mul.unlearned_model, mul.forget_test_loader)
            test_features, test_labels_orig = mia.extract_features(mul.unlearned_model, test_loader)

            # 保存三組特徵分布
            visualize_mia_features(
                retain_features, forget_features, test_features,
                output_dir=visualization_dir
            )

            print(f"[Debug] Before evaluate: X_test={X_test.shape}, y_test={len(y_test)}, labels={len(original_labels_test)}")

            if plot_mode == "eval":
                mia.plot_feature_distribution(
                    X_test, y_test, original_labels_test,
                    title="MIA Feature Distribution (Eval only)",
                    output_dir=visualization_dir
                )
            elif plot_mode == "all":
                # 畫 train+eval
                mia.plot_feature_distribution(
                    np.vstack([X_train, X_test]),
                    np.concatenate([y_train, y_test]),
                    np.concatenate([
                        np.full(len(y_train), -999),  # fake flag for train (避免 broadcast)
                        original_labels_test
                    ]),
                    title="MIA Feature Distribution (Train+Eval)",
                    output_dir=visualization_dir
                )

            return mia_results
        
        except Exception as e:
            logging.error("MIA評估失敗: %s", e)
            return {'error': str(e), 'accuracy': 0.5}
    
    def run_all_strategies(self, forget_classes_list, strategies=None):
        """運行所有WeightPruning策略"""
        if strategies is None:
            unlearner = WeightPruningUnlearner(None, self.device)
            strategies = unlearner.get_strategy_configs(prune_ratios=self.prune_ratios)

        # 創建實驗目錄
        exp_dir = self.storage.create_experiment_dir("weight_pruning")
        
        all_results = {}
        
        for forget_classes in forget_classes_list:
            forget_key = f"forget{len(forget_classes)}"
            all_results[forget_key] = {}
            
            for strategy_name, strategy_config in strategies.items():
                logging.info("處理策略: %s", strategy_name)
                
                try:
                    results = self.run_experiment(forget_classes, strategy_name, strategy_config)
                    all_results[forget_key][strategy_name] = results
                    
                    # 保存單個實驗結果
                    self.storage.save_results_to_json(
                        exp_dir, results, 
                        f"{forget_key}_{strategy_name}_results.json"
                    )
                    
                    logging.info("✓ %s 完成", strategy_name)
                    
                except Exception as e:
                    logging.exception("✗ %s 失敗: %s", strategy_name, e)
                    all_results[forget_key][strategy_name] = {'error': str(e)}
        
        # 使用統一的storage系統保存結果
        summary_csv, report_txt = self.storage.generate_comprehensive_summary(
            exp_dir, "WeightPruning", all_results
        )
        
        self.logger.info("WeightPruning實驗完成！")
        self.logger.info(f"結果目錄: {exp_dir}")
        self.logger.info(f"CSV摘要: {summary_csv}")
        self.logger.info(f"詳細報告: {report_txt}")
        
        return exp_dir, all_results
    
    def _generate_summary_csv(self, exp_dir, all_results):
        """生成總結CSV"""
        
        summary_data = []
        for forget_key, strategies in all_results.items():
            for strategy_name, results in strategies.items():
                if 'error' in results:
                    continue
                
                row = {
                    'forget_classes': forget_key,
                    'strategy': strategy_name,
                    'prune_ratio': results['strategy_config'].get('prune_ratio', 0),
                    'prune_strategy': results['strategy_config'].get('prune_strategy', 'unknown'),
                    'original_retain_acc': results['performance_metrics']['original_retain_acc'],
                    'unlearned_retain_acc': results['performance_metrics']['unlearned_retain_acc'],
                    'unlearned_forget_acc': results['performance_metrics']['unlearned_forget_acc'],
                    'gs_retain_acc': results['performance_metrics']['gs_retain_acc'],
                    'gs_forget_acc': results['performance_metrics']['gs_forget_acc'],
                    'zrf_score': results['unlearning_metrics']['zrf_score'],
                    'unlearning_time': results['timing']['unlearning_time'],
                    'cosine_similarity': results['enhanced_metrics'].get('cosine_similarity', 0),
                    'mia_accuracy': results['mia_results'].get('accuracy', 0.5)
                }
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        summary_path = f"{exp_dir}/results/weight_pruning_summary.csv"
        df.to_csv(summary_path, index=False, encoding='utf-8')
        
        return summary_path
    
    def _generate_report(self, exp_dir, all_results):
        """生成詳細報告"""
        report_path = f"{exp_dir}/results/weight_pruning_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("WeightPruning 遺忘實驗報告\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 實驗概述
            total_experiments = sum(len(strategies) for strategies in all_results.values())
            successful_experiments = sum(
                len([s for s in strategies.values() if 'error' not in s])
                for strategies in all_results.values()
            )
            
            f.write(f"實驗概述:\n")
            f.write(f"總實驗數: {total_experiments}\n")
            f.write(f"成功實驗數: {successful_experiments}\n")
            f.write(f"成功率: {100*successful_experiments/total_experiments:.1f}%\n\n")
            
            # 修剪策略分析
            pruning_strategies = set()
            for strategies in all_results.values():
                for strategy_name, results in strategies.items():
                    if 'error' not in results:
                        pruning_strategies.add(results['strategy_config'].get('prune_strategy', 'unknown'))
            
            f.write(f"測試的修剪策略: {', '.join(pruning_strategies)}\n")
            f.write(f"測試的修剪比例: 10%, 20%, 30%, 40%, 50%\n\n")
            
            # 各策略詳細結果
            for forget_key, strategies in all_results.items():
                f.write(f"\n{forget_key.upper()} 結果:\n")
                f.write("-"*50 + "\n")
                
                # 按ZRF分數排序
                strategy_scores = []
                for strategy_name, results in strategies.items():
                    if 'error' not in results:
                        zrf = results['unlearning_metrics']['zrf_score']
                        strategy_scores.append((strategy_name, zrf, results))
                
                strategy_scores.sort(key=lambda x: x[1], reverse=True)
                
                for i, (strategy_name, zrf, results) in enumerate(strategy_scores, 1):
                    f.write(f"{i}. {strategy_name}:\n")
                    f.write(f"   修剪策略: {results['strategy_config'].get('prune_strategy', 'unknown')}\n")
                    f.write(f"   修剪比例: {results['strategy_config'].get('prune_ratio', 0)*100:.0f}%\n")
                    f.write(f"   ZRF分數: {zrf:.4f}\n")
                    f.write(f"   保留準確率: {results['performance_metrics']['unlearned_retain_acc']:.2f}%\n")
                    f.write(f"   稀疏度增加: {results['model_analysis'].get('sparsity_increase', 0)*100:.1f}%\n")
                    f.write(f"   壓縮比: {results['model_analysis'].get('compression_ratio', 1):.3f}\n")
                    f.write(f"   訓練時間: {results['timing']['unlearning_time']:.1f}秒\n\n")
            
            # 修剪比例vs效果分析
            f.write("\n修剪比例效果分析:\n")
            f.write("-"*50 + "\n")
            
            ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
            for ratio in ratios:
                ratio_results = []
                for strategies in all_results.values():
                    for results in strategies.values():
                        if ('error' not in results and 
                            results['strategy_config'].get('prune_ratio') == ratio):
                            ratio_results.append(results['unlearning_metrics']['zrf_score'])
                
                if ratio_results:
                    avg_zrf = sum(ratio_results) / len(ratio_results)
                    f.write(f"修剪 {ratio*100:.0f}%: 平均ZRF = {avg_zrf:.4f} ({len(ratio_results)}個實驗)\n")
            
            # 修剪策略比較
            f.write("\n修剪策略比較:\n")
            f.write("-"*50 + "\n")
            
            strategy_results = {}
            for strategies in all_results.values():
                for results in strategies.values():
                    if 'error' not in results:
                        strategy_type = results['strategy_config'].get('prune_strategy', 'unknown')
                        if strategy_type not in strategy_results:
                            strategy_results[strategy_type] = []
                        strategy_results[strategy_type].append(results['unlearning_metrics']['zrf_score'])
            
            for strategy_type, scores in strategy_results.items():
                avg_score = sum(scores) / len(scores)
                f.write(f"{strategy_type}: 平均ZRF = {avg_score:.4f} ({len(scores)}個實驗)\n")
            
            f.write("\n" + "="*80 + "\n")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='WeightPruning 遺忘實驗')
    
    # 基礎參數
    parser.add_argument('--model_path', type=str, required=True, help='預訓練模型路徑')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='輸出目錄')
    parser.add_argument('--device', type=str, default='cuda', help='運算設備')
    
    # 批量實驗參數
    parser.add_argument('--forget_class_counts', nargs='+', type=int, default=[10, 20, 30], 
                       help='批量測試的遺忘類別數')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['magnitude_reset', 'magnitude_zero_lock', 'gradient', 'fisher', 'all'], 
                       default=['magnitude_reset'], help='測試策略')
    
    # 模型和訓練參數
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='數據載入線程數')
    parser.add_argument('--unlearn_epochs', type=int, default=20, help='修剪後微調的輪數 (fine_tune_epochs)')

    # 修剪比例
    parser.add_argument('--prune_ratios', nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help='修剪比例清單 (0~1 之間)，會套用到所有支援比例的策略')
    parser.add_argument('--prune_target_layers',
                        type=str,
                        default='head,late_blocks',
                        help='要修剪的層，逗號分隔。可用: head, late_blocks, all_blocks，或自訂子字串（例如 blocks.7.attn）'
    )
    
    # 黃金標準參數
    parser.add_argument("--use_forget_uniform", action="store_true",help="是否啟用忘記集高熵微調 (KL -> uniform)")
    parser.add_argument("--use_forget_distill", action="store_true",help="是否啟用忘記集蒸餾 (對齊 Gold)")
    parser.add_argument("--use_mixup", action="store_true",help="是否在 GS 訓練中使用 Mixup 資料增強")
    parser.add_argument("--use_cutmix", action="store_true",help="是否在 GS 訓練中使用 CutMix 資料增強")
    parser.add_argument('--mix_alpha', type=float, default=0.2, help='Mixup/CutMix 的 Beta 分布參數 alpha')
    parser.add_argument("--use_logit_penalty", action="store_true",help="是否在 GS 訓練中加入 logit 範數懲罰")
    parser.add_argument('--gs_epochs', type=int, default=250, help='黃金標準訓練輪數')
    parser.add_argument('--gs_lr', type=float, default=1e-3, help='黃金標準學習率')
    parser.add_argument('--gs_scheduler', type=str, default='onecycle', 
                       choices=['cosine', 'cosine_warmup', 'onecycle', 'step', 'plateau'], help='學習率調度器')
    parser.add_argument('--gs_min_lr', type=float, default=1e-6, help='最小學習率')
    parser.add_argument('--gs_weight_decay', type=float, default=0.05, help='權重衰減')
    
    parser.add_argument('--gs_use_ema', action='store_true', help='Use EMA during GS training/eval')
    parser.add_argument('--gs_ema_decay', type=float, default=0.999, help='EMA decay (beta) for GS')
    parser.add_argument('--gs_use_class_balance', action='store_true', help='Use class-balanced CE on retain set for GS')
    parser.add_argument('--gs_warmup_epochs', type=int, default=20, help='Warmup epochs for cosine_warmup scheduler')
    

    # 評估參數
    parser.add_argument('--run_mia', action='store_true', help='執行MIA評估')
    parser.add_argument('--no_mia', action='store_true', help='不執行MIA評估')
    parser.add_argument('--mia_epochs', type=int, default=30, help='MIA訓練輪數')
    parser.add_argument('--mia_lr', type=float, default=1e-3, help='MIA學習率')
    parser.add_argument('--run_enhanced_eval', action='store_true', default=True, help='執行增強評估')
    parser.add_argument('--no_enhanced_eval', action='store_true', help='不執行增強評估')

    # Cosine Head 參數
    parser.add_argument('--use_cosine_head', action='store_true', help='Use cosine classifier head for both GS/Unlearned')
    parser.add_argument('--cosine_scale', type=float, default=20.0, help='Scale s for cosine head logits')

    args = parser.parse_args()
    
    # 處理參數
    if args.no_mia:
        args.run_mia = False
    if args.no_enhanced_eval:
        args.run_enhanced_eval = False
    
    print("🚀 WeightPruning 批量遺忘實驗")
    print(f"📊 測試類別數: {args.forget_class_counts}")
    print(f"🔧 測試策略: {args.strategies}")
    print(f"📁 輸出目錄: {args.output_dir}")
    
    # 建立實驗實例
    experiment = WeightPruningExperiment(args)
    
    # 執行批量實驗
    exp_dir, all_results = experiment.run_all_experiments(args.forget_class_counts, args.strategies)
    
    print("✅ 所有實驗完成!")
    print(f"📊 結果保存在: {exp_dir}")
    print(f"📈 TensorBoard: tensorboard --logdir={experiment.tensorboard_dir}")


if __name__ == "__main__":
    main()
