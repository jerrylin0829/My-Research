"""
AttentionTargeted 遺忘實驗主程式
"""
import os
import sys
import time
import json
import argparse
import logging
import torch
import torch.nn as nn
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_LSA_classattn import VisionTransformer
from methods.attention_targeted_unlearner import AttentionTargetedUnlearner
# 改用 mul_vit_1 的實作（含 TensorBoard writer）
from mul_vit_1 import MembershipInferenceAttack, MachineUnlearning
from utils.evaluation_utils import enhanced_unlearning_evaluation, print_enhanced_evaluation_report, calculate_model_size_difference
from utils.storage_utils import ExperimentStorage


# 簡單 logging 設定
def setup_logging(log_file: str | None = None, level: int = logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=handlers,
    )


class AttentionTargetedExperiment:
    """AttentionTargeted 遺忘實驗管理器"""
    
    def __init__(
        self,
        base_model_path,
        output_dir: str = "./checkpoints",
        device: str | None = None,
        batch_size: int = 128,
        num_workers: int = 4,
        # GS 參數
        gs_epochs: int = 250,
        gs_lr: float = 1e-3,
        gs_scheduler: str = 'onecycle',
        gs_min_lr: float = 1e-6,
        gs_weight_decay: float = 0.05,
        # MIA 參數
        run_mia: bool = True,
        mia_epochs: int = 50,
        mia_lr: float = 1e-3,
        mia_use_scheduler: bool = True,
        # 增強評估
        run_enhanced_eval: bool = True,
    ):
        self.base_model_path = base_model_path
        self.storage = ExperimentStorage(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers
        # GS
        self.gs_epochs = gs_epochs
        self.gs_lr = gs_lr
        self.gs_scheduler = gs_scheduler
        self.gs_min_lr = gs_min_lr
        self.gs_weight_decay = gs_weight_decay
        # MIA
        self.run_mia = run_mia
        self.mia_epochs = mia_epochs
        self.mia_lr = mia_lr
        self.mia_use_scheduler = mia_use_scheduler
        # 增強評估
        self.run_enhanced_eval = run_enhanced_eval
        
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
    
    def run_experiment(self, forget_classes, strategy_name, strategy_config):
        """運行單個AttentionTargeted實驗"""
        logging.info("%s", "="*80)
        logging.info("開始AttentionTargeted實驗: %s", strategy_name)
        logging.info("遺忘類別: %s", sorted(forget_classes))
        logging.info("策略配置: %s", strategy_config)
        logging.info("%s", "="*80)
        
        # 初始化MachineUnlearning框架
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_output_dir = f"./temp_attention_{timestamp}"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # 建立 TensorBoard 目錄
        tb_dir = os.path.join(temp_output_dir, "gold_standard")
        os.makedirs(tb_dir, exist_ok=True)
        
        mul = MachineUnlearning(
            model_class=VisionTransformer,
            model_args=self.model_args,
            device=self.device,
            output_dir=temp_output_dir,
            log_dir=tb_dir
        )
        
        # 載入模型和準備數據
        mul.load_original_model(self.base_model_path)
        mul.prepare_data(dataset_name='CIFAR100', forget_classes=forget_classes)
        
        # 評估原始模型
        original_retain_acc = mul._evaluate_retain(mul.original_model)
        original_forget_acc = mul._evaluate_forget(mul.original_model)
        
        # 訓練黃金標準模型
        logging.info("訓練黃金標準模型...")
        mul.train_gold_standard(
            epochs=self.gs_epochs,
            lr=self.gs_lr,
            lr_scheduler_type=self.gs_scheduler,
            min_lr=self.gs_min_lr,
            weight_decay=self.gs_weight_decay,
        )
        gs_retain_acc = mul._evaluate_retain(mul.retrained_model)
        
        # 執行AttentionTargeted遺忘
        logging.info("執行AttentionTargeted遺忘: %s", strategy_name)
        start_time = time.time()
        
        unlearner = AttentionTargetedUnlearner(mul.unlearned_model, self.device)
        mul.unlearned_model = unlearner.unlearn(
            retain_loader=mul.retain_train_loader,
            forget_loader=mul.forget_train_loader,
            **strategy_config,
            class_mapping=mul.class_mapping,
            num_retain_classes=mul.num_retain_classes
        )
        
        unlearning_time = time.time() - start_time
        
        # 評估遺忘後的模型
        unlearned_retain_acc = mul._evaluate_retain(mul.unlearned_model)
        unlearned_forget_acc = mul._evaluate_forget(mul.unlearned_model)
        
        # 增強評估
        enhanced_results = {}
        if self.run_enhanced_eval:
            logging.info("執行增強評估...")
            enhanced_results = enhanced_unlearning_evaluation(
                mul.original_model, mul.unlearned_model, mul.retrained_model,
                mul.retain_test_loader, mul.forget_test_loader, mul.retain_test_loader, self.device
            )
        
        # 計算模型大小變化
        model_size_info = calculate_model_size_difference(mul.original_model, mul.unlearned_model)
        
        # MIA 評估（完整）
        mia_results = {'note': 'MIA disabled'}
        if self.run_mia:
            logging.info("執行MIA評估...")
            mia_results = self._run_mia(mul)
        
        # 整理結果
        results = {
            'experiment_info': {
                'method': 'AttentionTargeted',
                'strategy': strategy_name,
                'forget_classes': sorted(forget_classes),
                'num_forget_classes': len(forget_classes),
                'timestamp': timestamp
            },
            'strategy_config': strategy_config,
            'performance_metrics': {
                'original_retain_acc': original_retain_acc,
                'original_forget_acc': original_forget_acc,
                'unlearned_retain_acc': unlearned_retain_acc,
                'unlearned_forget_acc': unlearned_forget_acc,
                'gs_retain_acc': gs_retain_acc
            },
            'unlearning_metrics': {
                'zrf_score': enhanced_results.get('zrf_score', 0),
                'kl_divergence': enhanced_results.get('kl_divergence', 0),
                'forget_effectiveness': (original_forget_acc - (unlearned_forget_acc or 0)) / max(original_forget_acc, 1e-6) if original_forget_acc is not None else None,
                'retain_preservation': (unlearned_retain_acc / max(original_retain_acc, 1e-6)) if original_retain_acc is not None else None
            },
            'enhanced_metrics': enhanced_results,
            'model_analysis': model_size_info,
            'mia_results': mia_results,
            'timing': {
                'unlearning_time': unlearning_time,
                'retraining_time': mul.results.get('retraining_time', 0)
            }
        }
        
        # 注意：為保留 TensorBoard 日誌，不刪除臨時目錄 temp_output_dir
        # 可於需要時手動清理該目錄
        # import shutil
        # shutil.rmtree(temp_output_dir, ignore_errors=True)
        
        return results
    
    def _run_mia(self, mul):
        """運行完整的 MIA 評估 (mul_vit_1)"""
        try:
            mia = MembershipInferenceAttack(mul.unlearned_model, self.device)
            
            # 建立獨立測試資料（非成員）
            from torchvision import transforms, datasets
            from torch.utils.data import DataLoader
            
            test_dataset = datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            )
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            # 準備攻擊數據
            X_train, y_train, X_test, y_test, original_labels_test = mia.prepare_attack_data(
                mul.retain_test_loader, mul.forget_test_loader, test_loader
            )
            
            # 訓練攻擊模型
            mia.train_attack_model(
                X_train, y_train, X_test, y_test,
                epochs=self.mia_epochs, learning_rate=self.mia_lr, use_scheduler=self.mia_use_scheduler
            )
            
            # 評估攻擊效果
            return mia.evaluate_attack(X_test, y_test, original_labels_test)
        except Exception as e:
            logging.error("MIA評估失敗: %s", e)
            return {'error': str(e), 'accuracy': 0.5}
    
    def run_all_strategies(self, forget_classes_list, strategies=None):
        """運行所有AttentionTargeted策略"""
        if strategies is None:
            unlearner = AttentionTargetedUnlearner(None, self.device)
            strategies = unlearner.get_strategy_configs()
        
        # 創建實驗目錄
        exp_dir = self.storage.create_experiment_dir("attention_targeted")
        
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
        
        # 保存總結果
        self.storage.save_results_to_json(exp_dir, all_results, "attention_targeted_all_results.json")
        
        # 生成總結CSV
        self._generate_summary_csv(exp_dir, all_results)
        
        # 生成報告
        self._generate_report(exp_dir, all_results)
        
        logging.info("AttentionTargeted實驗完成！結果保存在: %s", exp_dir)
        return exp_dir, all_results
    
    def _generate_summary_csv(self, exp_dir, all_results):
        """生成總結CSV"""
        import pandas as pd
        
        summary_data = []
        for forget_key, strategies in all_results.items():
            for strategy_name, results in strategies.items():
                if 'error' in results:
                    continue
                
                row = {
                    'forget_classes': forget_key,
                    'strategy': strategy_name,
                    'attention_strategy': results['strategy_config'].get('attention_strategy', 'unknown'),
                    'epochs': results['strategy_config'].get('epochs', 0),
                    'lr': results['strategy_config'].get('lr', 0),
                    'original_retain_acc': results['performance_metrics']['original_retain_acc'],
                    'unlearned_retain_acc': results['performance_metrics']['unlearned_retain_acc'],
                    'unlearned_forget_acc': results['performance_metrics']['unlearned_forget_acc'],
                    'gs_retain_acc': results['performance_metrics']['gs_retain_acc'],
                    'zrf_score': results['unlearning_metrics']['zrf_score'],
                    'kl_divergence': results['unlearning_metrics']['kl_divergence'],
                    'forget_effectiveness': results['unlearning_metrics']['forget_effectiveness'],
                    'retain_preservation': results['unlearning_metrics']['retain_preservation'],
                    'unlearning_time': results['timing']['unlearning_time'],
                    'cosine_similarity': results['enhanced_metrics'].get('cosine_similarity', 0),
                    'activation_distance': results['enhanced_metrics'].get('activation_distance', 0),
                    'mia_accuracy': results['mia_results'].get('accuracy', 0.5)
                }
                summary_data.append(row)
        
        import pandas as pd
        df = pd.DataFrame(summary_data)
        summary_path = f"{exp_dir}/results/attention_targeted_summary.csv"
        df.to_csv(summary_path, index=False, encoding='utf-8')
        
        return summary_path
    
    def _generate_report(self, exp_dir, all_results):
        """生成詳細報告"""
        report_path = f"{exp_dir}/results/attention_targeted_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AttentionTargeted 遺忘實驗報告\n")
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
            
            # 注意力策略分析
            attention_strategies = set()
            for strategies in all_results.values():
                for strategy_name, results in strategies.items():
                    if 'error' not in results:
                        attention_strategies.add(results['strategy_config'].get('attention_strategy', 'unknown'))
            
            f.write(f"測試的注意力策略: {', '.join(attention_strategies)}\n\n")
            
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
                    f.write(f"   注意力策略: {results['strategy_config'].get('attention_strategy', 'unknown')}\n")
                    f.write(f"   訓練輪數: {results['strategy_config'].get('epochs', 0)}\n")
                    f.write(f"   學習率: {results['strategy_config'].get('lr', 0)}\n")
                    f.write(f"   ZRF分數: {zrf:.4f}\n")
                    f.write(f"   保留準確率: {results['performance_metrics']['unlearned_retain_acc']:.2f}%\n")
                    f.write(f"   遺忘準確率: {results['performance_metrics']['unlearned_forget_acc']:.2f}%\n")
                    f.write(f"   訓練時間: {results['timing']['unlearning_time']:.1f}秒\n")
                    f.write(f"   餘弦相似度: {results['enhanced_metrics'].get('cosine_similarity', 0):.3f}\n\n")
            
            # 策略比較
            f.write("\n注意力策略比較:\n")
            f.write("-"*50 + "\n")
            
            strategy_results = {}
            for strategies in all_results.values():
                for results in strategies.values():
                    if 'error' not in results:
                        strategy_type = results['strategy_config'].get('attention_strategy', 'unknown')
                        if strategy_type not in strategy_results:
                            strategy_results[strategy_type] = []
                        strategy_results[strategy_type].append(results['unlearning_metrics']['zrf_score'])
            
            for strategy_type, scores in strategy_results.items():
                avg_score = sum(scores) / len(scores)
                f.write(f"{strategy_type}: 平均ZRF = {avg_score:.4f} ({len(scores)}個實驗)\n")
            
            f.write("\n" + "="*80 + "\n")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='AttentionTargeted 遺忘實驗')
    parser.add_argument("--model_path", type=str, 
                       default="/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth",
                       help="預訓練模型路徑")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="輸出目錄")
    parser.add_argument("--device", type=str, default=None, help="運算裝置，如 cuda 或 cpu，預設自動偵測")
    parser.add_argument("--batch_size", type=int, default=128, help="測試/MIA 批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers 數")
    # GS 參數
    parser.add_argument("--gs_epochs", type=int, default=250)
    parser.add_argument("--gs_lr", type=float, default=1e-3)
    parser.add_argument("--gs_scheduler", type=str, choices=['cosine', 'onecycle', 'step', 'plateau'], default='onecycle')
    parser.add_argument("--gs_min_lr", type=float, default=1e-6)
    parser.add_argument("--gs_weight_decay", type=float, default=0.05)
    # MIA 參數
    parser.add_argument("--run_mia", action="store_true", default=True, help="是否執行 MIA")
    parser.add_argument("--no_mia", dest="run_mia", action="store_false", help="停用 MIA")
    parser.add_argument("--mia_epochs", type=int, default=50)
    parser.add_argument("--mia_lr", type=float, default=1e-3)
    parser.add_argument("--mia_use_scheduler", action="store_true", default=True)
    parser.add_argument("--no_mia_scheduler", dest="mia_use_scheduler", action="store_false")
    # 增強評估
    parser.add_argument("--run_enhanced_eval", action="store_true", default=True)
    parser.add_argument("--no_enhanced_eval", dest="run_enhanced_eval", action="store_false")
    # 遺忘與策略
    parser.add_argument("--forget_classes", type=str, nargs='+', 
                       default=["17,24,39,53,56,68,75,76,82,94", "0,1,2,3,4,5,6,7,8,9", "10,11,12,13,14,15,16,17,18,19"],
                       help="要遺忘的類別組合。可用 'a-b' 或 'a,b,c' 形式，支援多組")
    parser.add_argument("--strategies", type=str, nargs='+',
                       choices=['head_specific', 'layer_specific', 'pattern_based', 'all'],
                       default=['all'],
                       help="要測試的注意力策略")
    # logging
    parser.add_argument("--log_file", type=str, default=None, help="log 檔輸出位置（預設 logs/attention_*.log）")
    
    args = parser.parse_args()
    
    # 解析遺忘類別
    forget_classes_list = []
    for classes_str in args.forget_classes:
        if '-' in classes_str:
            start, end = map(int, classes_str.split('-'))
            forget_classes = set(range(start, end + 1))
        else:
            forget_classes = {int(x.strip()) for x in classes_str.split(',') if x.strip()}
        forget_classes_list.append(forget_classes)
    
    # 設定 logging
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    default_log = args.log_file or os.path.join('logs', f'attention_{timestamp}.log')
    setup_logging(default_log)
    logging.info("AttentionTargeted 實驗啟動")
    logging.info("Arguments: %s", vars(args))
    
    # 建立實驗器
    experiment = AttentionTargetedExperiment(
        base_model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gs_epochs=args.gs_epochs,
        gs_lr=args.gs_lr,
        gs_scheduler=args.gs_scheduler,
        gs_min_lr=args.gs_min_lr,
        gs_weight_decay=args.gs_weight_decay,
        run_mia=args.run_mia,
        mia_epochs=args.mia_epochs,
        mia_lr=args.mia_lr,
        mia_use_scheduler=args.mia_use_scheduler,
        run_enhanced_eval=args.run_enhanced_eval,
    )
    
    # 選擇策略
    if 'all' in args.strategies:
        strategies = None  # 使用所有策略
    else:
        # 根據選擇的策略篩選
        unlearner = AttentionTargetedUnlearner(None, experiment.device)
        all_strategies = unlearner.get_strategy_configs()
        strategies = {
            name: config for name, config in all_strategies.items()
            if config.get('attention_strategy') in args.strategies
        }
    
    # 運行實驗
    exp_dir, results = experiment.run_all_strategies(forget_classes_list, strategies)
    
    logging.info("所有AttentionTargeted實驗完成！")
    logging.info("結果保存在: %s", exp_dir)


if __name__ == "__main__":
    main()
