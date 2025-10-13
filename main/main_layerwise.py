"""
LayerWise 遺忘實驗主程式
完整版本 - 使用簡潔日誌系統
"""
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import pandas as pd
import shutil
import random
import numpy as np
import logging
from datetime import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_LSA_classattn import VisionTransformer
from methods.layerwise_unlearner import LayerWiseUnlearner
from utils.evaluation_utils import enhanced_unlearning_evaluation, calculate_model_size_difference
from utils.storage_utils import ExperimentStorage
from mul_vit_1 import MachineUnlearning, MembershipInferenceAttack

def generate_forget_classes(num_classes, total_classes=100, seed=42):
    """
    生成固定的遺忘類別列表
    
    Args:
        num_classes: 要遺忘的類別數量
        total_classes: 總類別數
        seed: 隨機種子
    
    Returns:
        set: 遺忘類別的集合
    """
    random.seed(seed + num_classes)
    np.random.seed(seed + num_classes)
    
    forget_classes = set(random.sample(range(total_classes), num_classes))
    return forget_classes


class LayerWiseExperiment:
    """LayerWise 遺忘實驗管理器 - 使用統一的Storage系統"""
    
    def __init__(self, args):
        self.args = args
        self.base_model_path = args.model_path
        self.device = args.device if hasattr(args, 'device') else 'cuda'
        
        # 使用統一的儲存系統
        self.storage = ExperimentStorage(args.output_dir)
        
        # 創建完整實驗結構
        self.exp_dir, self.exp_name = self.storage.create_complete_experiment_structure("layerwise")
        
        # 設置日誌系統
        self.logger, self.log_file = self.storage.setup_logging_system(self.exp_dir, self.exp_name)
        
        # 設置TensorBoard
        self.tensorboard_dir, self.main_writer = self.storage.setup_tensorboard(self.exp_dir)
        
        self.logger.info(f"TensorBoard目錄: {self.tensorboard_dir}")
        self.logger.info("使用以下命令查看TensorBoard:")
        self.logger.info(f"  tensorboard --logdir={self.tensorboard_dir}")
    
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
            self.exp_dir, "LayerWise", args, self.model_args
        )
        
        # 統一的實驗開始日誌
        self.storage.log_experiment_start(
            self.logger, "LayerWise", self.exp_name, self.exp_dir, self.config, args
        )
        
        print(f"🚀 LayerWise實驗開始 - 目錄: {self.exp_dir}")
    def run_single_experiment(self, forget_classes, strategy_name, strategy_config):
        """運行單個LayerWise實驗"""
        experiment_start_time = time.time()
        
        # 為每個實驗創建獨立的writer
        exp_id = f"forget{len(forget_classes)}_{strategy_name}"
        run_tb_dir = os.path.join(self.tensorboard_dir, exp_id)
        os.makedirs(run_tb_dir, exist_ok=True)
        writer = SummaryWriter(run_tb_dir)

        # 實驗開始記錄
        self.logger.info("\n" + "="*60)
        self.logger.info(f"開始實驗: {strategy_name}")
        self.logger.info(f"遺忘類別數量: {len(forget_classes)}")
        self.logger.info(f"遺忘類別: {sorted(forget_classes)}")
        self.logger.info(f"策略配置: {strategy_config}")
        self.logger.info("="*60)
        
        print(f"\n🎯 執行 {strategy_name} (遺忘{len(forget_classes)}類)")
        
        # 使用臨時目錄
        temp_dir = f"./temp_layerwise_{strategy_name}_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # === 階段1: 初始化 ===
            self.logger.info("階段1: 初始化實驗框架")
            
            # 建立 GS 的 TensorBoard 目錄
            gold_tb_dir = os.path.join(run_tb_dir, "gold_standard")
            os.makedirs(gold_tb_dir, exist_ok=True)
            
            mul = MachineUnlearning(
                model_class=VisionTransformer,
                model_args=self.model_args,
                device=self.device,
                output_dir=temp_dir,
                log_dir=gold_tb_dir  # 直接指定 GS 訓練曲線輸出位置
            )
            
            # === 階段2: 載入模型和數據 ===
            self.logger.info("階段2: 載入模型和準備數據")
            mul.load_original_model(self.base_model_path)
            mul.prepare_data(dataset_name='CIFAR100', forget_classes=forget_classes)
            
            self.logger.info(f"保留訓練集大小: {len(mul.retain_train_loader.dataset)}")
            self.logger.info(f"遺忘訓練集大小: {len(mul.forget_train_loader.dataset)}")
            
            # === 階段3: 評估原始模型 ===
            self.logger.info("階段3: 評估原始模型性能")
            original_retain_acc = mul._evaluate_retain(mul.original_model)
            original_forget_acc = mul._evaluate_forget(mul.original_model)
            
            self.logger.info(f"原始模型 - 保留準確率: {original_retain_acc:.4f}%")
            self.logger.info(f"原始模型 - 遺忘準確率: {original_forget_acc:.4f}%")
            
            print(f"📊 原始: 保留 {original_retain_acc:.2f}% | 遺忘 {original_forget_acc:.2f}%")
            
            # === 階段4: 訓練黃金標準 ===
            if not hasattr(mul, 'retrained_model') or mul.retrained_model is None:
                self.logger.info("階段4: 訓練黃金標準模型")
                print("🏆 訓練黃金標準模型...")

                mul.train_gold_standard(
                    epochs=self.args.gs_epochs,
                    lr=self.args.gs_lr,
                    lr_scheduler_type=self.args.gs_lr_scheduler,
                    weight_decay=self.args.gs_weight_decay,
                )
                
            else:
                self.logger.info("階段4: 使用既有黃金標準模型")
            
            gs_retain_acc = mul._evaluate_retain(mul.retrained_model)
            self.logger.info(f"黃金標準模型 - 保留準確率: {gs_retain_acc:.4f}%")
            print(f"🏆 黃金標準: 保留 {gs_retain_acc:.2f}%")
            
            # === 階段5: 執行LayerWise遺忘 ===
            self.logger.info("階段5: 執行LayerWise遺忘")
            unlearn_start = time.time()
            
            print(f"🧠 執行LayerWise遺忘 ({strategy_name})...")
            
            unlearner = LayerWiseUnlearner(mul.unlearned_model, self.device)
            
            # 準備遺忘參數
            unlearn_params = {
                'retain_loader': mul.retain_train_loader,
                'forget_loader': mul.forget_train_loader,
                'epochs': self.args.unlearn_epochs,
                'lr': self.args.unlearn_lr,
                # 'lr_scheduler_type': self.args.unlearn_lr_scheduler,
                'class_mapping': mul.class_mapping,
                'num_retain_classes': mul.num_retain_classes,
                **strategy_config
            }
            
            mul.unlearned_model = unlearner.unlearn(**unlearn_params)
            unlearning_time = time.time() - unlearn_start
            
            self.logger.info(f"LayerWise遺忘完成，用時: {unlearning_time:.2f}秒")
            
            # === 階段6: 評估遺忘模型 ===
            self.logger.info("階段6: 評估遺忘後模型")
            
            unlearned_retain_acc = mul._evaluate_retain(mul.unlearned_model)
            unlearned_forget_acc = mul._evaluate_forget(mul.unlearned_model)
            
            self.logger.info(f"遺忘模型 - 保留準確率: {unlearned_retain_acc:.4f}%")
            self.logger.info(f"遺忘模型 - 遺忘準確率: {unlearned_forget_acc:.4f}%")
            
            print(f"📊 遺忘後: 保留 {unlearned_retain_acc:.2f}% | 遺忘 {unlearned_forget_acc:.2f}%")
            
            # 計算關鍵指標
            retain_preservation = unlearned_retain_acc / max(original_retain_acc, 1e-6)
            forget_effectiveness = (original_forget_acc - (unlearned_forget_acc or 0)) / max(original_forget_acc, 1e-6)
            
            self.logger.info(f"保留能力保持率: {retain_preservation:.4f}")
            self.logger.info(f"遺忘有效性: {forget_effectiveness:.4f}")
            
            # 寫入基本摘要到 run 根目錄
            writer.add_scalar("Original/Retain_Acc", original_retain_acc, 0)
            writer.add_scalar("Original/Forget_Acc", original_forget_acc, 0)
            writer.add_scalar("Unlearned/Retain_Acc", unlearned_retain_acc, 0)
            writer.add_scalar("Unlearned/Forget_Acc", unlearned_forget_acc, 0)
            writer.add_scalar("GoldStandard/Retain_Acc", gs_retain_acc, 0)
            
            # === 階段7: 增強評估 ===
            enhanced_results = {}
            if self.args.run_enhanced_eval:
                self.logger.info("階段7: 執行增強評估")
                print("📈 執行增強評估...")
                
                enhanced_results = enhanced_unlearning_evaluation(
                    mul.original_model, mul.unlearned_model, mul.retrained_model,
                    mul.retain_test_loader, mul.forget_test_loader,
                    mul.retain_test_loader, self.device
                )
                
                # 記錄詳細的增強評估結果
                self.logger.info("增強評估結果:")
                for metric, value in enhanced_results.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {metric}: {value:.6f}")
                
                zrf_score = enhanced_results.get('zrf_score', 0)
                self.logger.info(f"ZRF分數: {zrf_score:.6f}")
                print(f"🎯 ZRF分數: {zrf_score:.4f}")
                
                # 寫入增強評估標量
                writer.add_scalar("Unlearning/ZRF", zrf_score, 0)
                writer.add_scalar("Unlearning/KL", enhanced_results.get('kl_divergence', 0.0), 0)
            
            # === 階段8: 完整MIA評估 ===
            mia_results = {}
            if self.args.run_mia:
                self.logger.info("階段8: 執行完整MIA評估")
                print("🕵️ 執行MIA評估...")
                
                try:
                    mia_results = self._run_comprehensive_mia(mul)
                    
                    # 記錄詳細MIA結果
                    self.logger.info("MIA評估結果:")
                    for metric, value in mia_results.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"  MIA {metric}: {value:.6f}")
                    
                    mia_acc = mia_results.get('accuracy', 0.5)
                    print(f"🕵️ MIA準確率: {mia_acc:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"MIA評估失敗: {e}")
                    self.logger.debug("MIA錯誤詳情:", exc_info=True)
                    mia_results = {'error': str(e), 'accuracy': 0.5}
            
            # === 計算模型分析 ===
            model_size_info = {}
            try:
                model_size_info = calculate_model_size_difference(
                    mul.original_model, mul.unlearned_model
                )
                self.logger.info("模型大小分析:")
                for key, value in model_size_info.items():
                    self.logger.info(f"  {key}: {value}")
            except Exception as e:
                self.logger.warning(f"模型分析失敗: {e}")
            
            # === 整理完整結果 ===
            total_experiment_time = time.time() - experiment_start_time
            
            results = {
                'experiment_info': {
                    'method': 'LayerWise',
                    'strategy': strategy_name,
                    'forget_classes': sorted(forget_classes),
                    'num_forget_classes': len(forget_classes),
                    'timestamp': datetime.now().isoformat(),
                    'total_experiment_time': total_experiment_time
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
                    'forget_effectiveness': forget_effectiveness,
                    'retain_preservation': retain_preservation
                },
                'enhanced_metrics': enhanced_results,
                'model_analysis': model_size_info,
                'mia_results': mia_results,
                'timing': {
                    'unlearning_time': unlearning_time,
                    'total_experiment_time': total_experiment_time
                }
            }
            
            # 保存模型（如果需要）
            if self.args.save_models:
                self._save_models(mul, strategy_name, len(forget_classes))
            
            # 記錄實驗完成
            self.logger.info("\n" + "="*60)
            self.logger.info(f"實驗 {strategy_name} 成功完成！")
            self.logger.info(f"總用時: {total_experiment_time:.2f}秒")
            self.logger.info(f"ZRF分數: {enhanced_results.get('zrf_score', 0):.6f}")
            self.logger.info("="*60)
            
            print(f"✅ {strategy_name} 完成 - ZRF: {enhanced_results.get('zrf_score', 0):.4f}")
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"實驗 {strategy_name} 失敗: {error_msg}")
            self.logger.error("詳細錯誤信息:", exc_info=True)
            print(f"❌ {strategy_name} 失敗: {error_msg}")
            return {'error': error_msg, 'strategy': strategy_name}
        
        finally:
            # 清理臨時目錄
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _run_comprehensive_mia(self, mul):
        """運行完整的MIA評估"""
        self.logger.info("開始完整MIA評估流程...")
        
        try:
            # 創建MIA攻擊器
            mia = MembershipInferenceAttack(
                target_model=mul.unlearned_model,
                device=self.device
            )
            
            self.logger.info("創建CIFAR100測試數據集...")
            # 創建完整的測試數據集
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            
            test_dataset = datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.args.batch_size, 
                shuffle=False, 
                num_workers=getattr(self.args, 'num_workers', 4)
            )
            
            self.logger.info("準備MIA攻擊數據...")
            # 準備攻擊數據
            X_train, y_train, X_test, y_test, original_labels_test = mia.prepare_attack_data(
                mul.retain_test_loader, 
                mul.forget_test_loader, 
                test_loader,
            )
            
            # 可視化並保存特徵分布圖
            mia.plot_feature_distribution(
                np.vstack([X_train, X_test]), 
                np.concatenate([y_train, y_test]), 
                title="MIA Feature Distribution", 
                output_dir=self.exp_dir
            )

            self.logger.info(f"MIA訓練集大小: {len(X_train)}")
            self.logger.info(f"MIA測試集大小: {len(X_test)}")
            
            # 訓練MIA攻擊模型
            self.logger.info("訓練MIA攻擊模型...")
            mia_epochs = self.args.mia_epochs
            mia_lr = getattr(self.args, 'mia_lr', 1e-3)
            use_scheduler = getattr(self.args, 'mia_use_scheduler', True)
            
            mia.train_attack_model(
                X_train, y_train, X_test, y_test, 
                epochs=mia_epochs,
                learning_rate=mia_lr,
                use_scheduler=use_scheduler
            )
            
            self.logger.info("評估MIA攻擊效果...")
            # 評估攻擊效果
            mia_results = mia.evaluate_attack(X_test, y_test, original_labels_test)
            
            self.logger.info("MIA評估完成")
            return mia_results
            
        except Exception as e:
            self.logger.error(f"MIA評估過程中發生錯誤: {e}")
            self.logger.debug("MIA錯誤詳細堆棧:", exc_info=True)
            
            # 返回默認值而不是崩潰
            return {
                'error': str(e),
                'accuracy': 0.5,
                'retain_acc': 0.5,
                'forget_acc': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5
            }
    
    def _save_models(self, mul, strategy_name, num_forget_classes):
        """保存模型"""
        model_dir = f"{self.exp_dir}/models/forget{num_forget_classes}_{strategy_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存各個模型
        torch.save(mul.unlearned_model.state_dict(), 
                   f"{model_dir}/unlearned_model.pth")
        
        if hasattr(mul, 'retrained_model') and mul.retrained_model is not None:
            torch.save(mul.retrained_model.state_dict(), 
                       f"{model_dir}/gold_standard_model.pth")
        
        self.logger.info(f"模型已保存到: {model_dir}")
    
    def run_all_experiments(self, forget_classes_configs):
        """運行所有LayerWise實驗"""
        # 獲取策略配置
        temp_unlearner = LayerWiseUnlearner(None, self.device)
        strategies = temp_unlearner.get_strategy_configs()
        
        # 篩選策略
        if hasattr(self.args, 'strategies') and 'all' not in self.args.strategies:
            strategies = {name: config for name, config in strategies.items()
                         if name in self.args.strategies}
        
        total_experiments = len(forget_classes_configs) * len(strategies)
        
        # 詳細的實驗計劃記錄
        self.logger.info("\n" + "="*80)
        self.logger.info("實驗執行計劃")
        self.logger.info("="*80)
        self.logger.info(f"總實驗數量: {total_experiments}")
        self.logger.info(f"遺忘類別配置: {len(forget_classes_configs)} 種")
        self.logger.info(f"LayerWise策略: {len(strategies)} 種")
        
        self.logger.info("\n遺忘類別詳情:")
        for config in forget_classes_configs:
            self.logger.info(f"  {config['num_classes']}類: {sorted(config['classes'])}")
        
        self.logger.info(f"\nLayerWise策略詳情:")
        for name, config in strategies.items():
            self.logger.info(f"  {name}: {config}")
        
        self.logger.info("="*80)
        
        # 控制台輸出計劃摘要
        print(f"\n📋 實驗計劃: {total_experiments} 個實驗")
        print(f"🎯 類別配置: {[c['num_classes'] for c in forget_classes_configs]}")
        print(f"🔧 策略: {list(strategies.keys())}")
        
        all_results = {}
        experiment_count = 0
        successful_experiments = 0
        failed_experiments = 0
        
        # 使用進度條
        with tqdm(total=total_experiments, desc="LayerWise實驗總進度", position=0) as main_pbar:
            
            for config_idx, config in enumerate(forget_classes_configs, 1):
                num_classes = config['num_classes']
                forget_classes = config['classes']
                forget_key = f"forget{num_classes}"
                all_results[forget_key] = {}
                
                self.logger.info(f"\n處理遺忘配置 {config_idx}/{len(forget_classes_configs)}: {num_classes} 個類別")
                self.logger.info(f"遺忘類別: {sorted(forget_classes)}")
                
                print(f"\n📁 處理 {forget_key} ({config_idx}/{len(forget_classes_configs)})")
                
                # 保存當前配置的詳細參數
                args_path = f"{self.exp_dir}/configs/{forget_key}_experiment_args.txt"
                self._save_detailed_args_file(args_path, config, strategies)
                
                for strategy_name, strategy_config in strategies.items():
                    experiment_count += 1
                    
                    # 更新進度條描述
                    main_pbar.set_description(f"執行 {strategy_name} (忘記{num_classes}類) [{experiment_count}/{total_experiments}]")
                    
                    try:
                        # 執行單個實驗
                        results = self.run_single_experiment(
                            forget_classes, strategy_name, strategy_config
                        )
                        
                        all_results[forget_key][strategy_name] = results
                        
                        # 保存單個實驗結果
                        result_file = f"{self.exp_dir}/results/{forget_key}_{strategy_name}_detailed.json"
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        
                        # 更新統計
                        if 'error' not in results:
                            successful_experiments += 1
                            zrf = results['unlearning_metrics']['zrf_score']
                            
                            # 更新進度條後綴
                            main_pbar.set_postfix({
                                'ZRF': f"{zrf:.3f}",
                                'Success': f"{successful_experiments}/{experiment_count}"
                            })
                            
                            self.logger.info(f"✓ {strategy_name} 成功完成 - ZRF: {zrf:.6f}")
                        else:
                            failed_experiments += 1
                            main_pbar.set_postfix({
                                'Success': f"{successful_experiments}/{experiment_count}",
                                'Failed': failed_experiments
                            })
                            self.logger.error(f"✗ {strategy_name} 失敗: {results['error']}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        all_results[forget_key][strategy_name] = {'error': error_msg}
                        failed_experiments += 1
                        
                        self.logger.error(f"✗ {strategy_name} 異常: {error_msg}")
                        self.logger.error("詳細異常信息:", exc_info=True)
                        print(f"❌ {strategy_name} 異常")
                    
                    # 更新主進度條
                    main_pbar.update(1)
                
                # 當前配置完成摘要
                self.logger.info(f"\n{forget_key} 配置完成:")
                self.logger.info(f"  成功: {sum(1 for r in all_results[forget_key].values() if 'error' not in r)}/{len(strategies)}")
                self.logger.info(f"  失敗: {sum(1 for r in all_results[forget_key].values() if 'error' in r)}/{len(strategies)}")
        
        # 實驗完成總結
        self.logger.info("\n" + "="*80)
        self.logger.info("所有實驗完成！")
        self.logger.info("="*80)
        self.logger.info(f"總實驗數: {experiment_count}")
        self.logger.info(f"成功實驗: {successful_experiments}")
        self.logger.info(f"失敗實驗: {failed_experiments}")
        self.logger.info(f"成功率: {successful_experiments/max(experiment_count,1)*100:.2f}%")
        
        # 使用統一的storage系統生成報告
        self.logger.info("\n生成實驗報告...")
        summary_csv, report_txt = self.storage.generate_comprehensive_summary(
            self.exp_dir, "LayerWise", all_results
        )
        
        self.logger.info(f"CSV總結報告: {summary_csv}")
        self.logger.info(f"詳細文字報告: {report_txt}")
        self.logger.info("="*80)
        
        # 控制台最終摘要
        print(f"\n🎉 LayerWise實驗全部完成！")
        print(f"📊 成功率: {successful_experiments}/{experiment_count} ({successful_experiments/max(experiment_count,1)*100:.1f}%)")
        print(f"📁 結果目錄: {self.exp_dir}")
        
        return self.exp_dir, all_results
    
    def _save_detailed_args_file(self, args_path, config, strategies):
        """保存詳細的參數文件"""
        with open(args_path, 'w', encoding='utf-8') as f:
            f.write(f"# LayerWise遺忘實驗詳細參數\n")
            f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 實驗目錄: {self.exp_dir}\n\n")
            
            f.write("=== 基本信息 ===\n")
            f.write(f"method: LayerWise\n")
            f.write(f"model_path: {self.base_model_path}\n")
            f.write(f"device: {self.device}\n")
            f.write(f"experiment_name: {self.exp_name}\n\n")
            
            f.write("=== 遺忘配置 ===\n")
            f.write(f"num_forget_classes: {config['num_classes']}\n")
            f.write(f"forget_classes: {','.join(map(str, sorted(config['classes'])))}\n")
            f.write(f"seed: {config['seed']}\n\n")
            
            f.write("=== LayerWise策略 ===\n")
            for name, strategy_config in strategies.items():
                f.write(f"{name}: {strategy_config}\n")
            f.write("\n")
            
            f.write("=== 訓練參數 ===\n")
            f.write(f"unlearn_epochs: {self.args.unlearn_epochs}\n")
            f.write(f"unlearn_lr: {self.args.unlearn_lr}\n")
            f.write(f"batch_size: {self.args.batch_size}\n\n")
            
            f.write("=== MIA參數 ===\n")
            f.write(f"run_mia: {self.args.run_mia}\n")
            f.write(f"mia_epochs: {self.args.mia_epochs}\n")
            f.write(f"mia_lr: {getattr(self.args, 'mia_lr', 1e-3)}\n")
    
    def _generate_comprehensive_csv(self, all_results):
        """生成詳細的CSV報告"""
        summary_data = []
        
        for forget_key, strategies in all_results.items():
            for strategy_name, results in strategies.items():
                if 'error' in results:
                    continue
                
                row = {
                    'forget_classes': forget_key,
                    'strategy': strategy_name,
                    'num_forget_classes': results['experiment_info']['num_forget_classes'],
                    'forget_classes_list': ','.join(map(str, results['experiment_info']['forget_classes'])),
                    'original_retain_acc': results['performance_metrics']['original_retain_acc'],
                    'unlearned_retain_acc': results['performance_metrics']['unlearned_retain_acc'],
                    'unlearned_forget_acc': results['performance_metrics']['unlearned_forget_acc'],
                    'gs_retain_acc': results['performance_metrics']['gs_retain_acc'],
                    'zrf_score': results['unlearning_metrics']['zrf_score'],
                    'kl_divergence': results['unlearning_metrics']['kl_divergence'],
                    'forget_effectiveness': results['unlearning_metrics']['forget_effectiveness'],
                    'retain_preservation': results['unlearning_metrics']['retain_preservation'],
                    'unlearning_time': results['timing']['unlearning_time'],
                    'mia_accuracy': results['mia_results'].get('accuracy', 0.5),
                    'timestamp': results['experiment_info']['timestamp']
                }
                
                # 添加增強評估的詳細指標
                enhanced = results.get('enhanced_metrics', {})
                for metric, value in enhanced.items():
                    if isinstance(value, (int, float)):
                        row[f'enhanced_{metric}'] = value
                
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = f"{self.exp_dir}/results/layerwise_comprehensive_summary.csv"
            df.to_csv(summary_path, index=False, encoding='utf-8')
            
            self.logger.info(f"詳細CSV報告已保存: {summary_path}")
            return summary_path
        else:
            self.logger.warning("沒有實驗結果可以生成CSV")
            return None
    
    def _generate_detailed_report(self, all_results):
        """生成詳細的文字報告"""
        report_path = f"{self.exp_dir}/results/layerwise_detailed_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LayerWise 遺忘實驗詳細報告\n")
            f.write("="*80 + "\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"實驗目錄: {self.exp_dir}\n\n")
            
            # 各遺忘類別配置的詳細結果
            for forget_key, strategies in all_results.items():
                f.write(f"{forget_key.upper()} 詳細結果:\n")
                f.write("-" * 50 + "\n")
                
                # 統計成功和失敗
                strategy_success = sum(1 for r in strategies.values() if 'error' not in r)
                strategy_total = len(strategies)
                
                f.write(f"策略執行: {strategy_success}/{strategy_total} 成功\n\n")
                
                # 按ZRF分數排序
                successful_strategies = []
                failed_strategies = []
                
                for strategy_name, results in strategies.items():
                    if 'error' not in results:
                        zrf = results['unlearning_metrics']['zrf_score']
                        mia_acc = results['mia_results'].get('accuracy', 0.5)
                        successful_strategies.append((strategy_name, zrf, mia_acc, results))
                    else:
                        failed_strategies.append((strategy_name, results['error']))
                
                successful_strategies.sort(key=lambda x: x[1], reverse=True)
                
                # 成功的策略排名
                if successful_strategies:
                    f.write("成功策略排名 (按ZRF分數):\n")
                    for i, (strategy_name, zrf, mia_acc, results) in enumerate(successful_strategies, 1):
                        retain_acc = results['performance_metrics']['unlearned_retain_acc']
                        forget_acc = results['performance_metrics']['unlearned_forget_acc']
                        time_cost = results['timing']['unlearning_time']
                        
                        f.write(f"{i:2d}. {strategy_name:20s} | ZRF: {zrf:8.6f} | ")
                        f.write(f"MIA: {mia_acc:6.4f} | 保留: {retain_acc:6.2f}% | ")
                        f.write(f"遺忘: {forget_acc:6.2f}% | 時間: {time_cost:6.1f}s\n")
                    f.write("\n")
                
                # 失敗的策略
                if failed_strategies:
                    f.write("失敗策略:\n")
                    for strategy_name, error in failed_strategies:
                        f.write(f"✗ {strategy_name}: {error}\n")
                    f.write("\n")
                
                f.write("\n")
        
        self.logger.info(f"詳細文字報告已保存: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='LayerWise 遺忘實驗 - 完整版')
    
    # === 基本設定 ===
    parser.add_argument("--model_path", type=str, 
                       default="/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth",
                       help="預訓練模型路徑")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="輸出目錄")
    parser.add_argument("--device", type=str, default="cuda", help="計算設備")
    
    # === 遺忘類別設定 ===
    parser.add_argument("--forget_class_counts", type=int, nargs='+', 
                       default=[10, 20, 30, 40, 50],
                       help="要遺忘的類別數量列表")
    parser.add_argument("--seed", type=int, default=42,
                       help="隨機種子，確保每次實驗類別組合一致")
    parser.add_argument("--forget_classes", type=str, default=None,
                       help="直接指定遺忘類別 (e.g., '20-29' or '20,21,22')，會覆蓋forget_class_counts")
    
    # === LayerWise 策略參數 ===
    parser.add_argument("--strategies", type=str, nargs='+',
                       choices=['head_only', 'head_plus_last1', 'head_plus_last2', 
                               'head_plus_last3', 'head_plus_last4', 'head_plus_last5', 
                               'progressive', 'all'],
                       default=['all'],
                       help="要測試的LayerWise策略")
    
    # === 訓練參數 ===
    parser.add_argument("--unlearn_epochs", type=int, default=30, 
                       help="LayerWise遺忘訓練輪數")
    parser.add_argument("--unlearn_lr", type=float, default=5e-5, 
                       help="LayerWise遺忘學習率")
    parser.add_argument("--unlearn_lr_min", type=float, default=1e-6, 
                       help="LayerWise遺忘最小學習率")
    # parser.add_argument("--unlearn_lr_scheduler", type=str, 
    #                    choices=['cosine', 'step', 'plateau', 'onecycle'],
    #                    default='cosine', help="LayerWise遺忘學習率調度器")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                       help="權重衰減")
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="數據加載器工作進程數")
    
    # === 黃金標準模型參數 ===
    parser.add_argument("--gs_epochs", type=int, default=250, 
                       help="黃金標準模型訓練輪數")
    parser.add_argument("--gs_lr", type=float, default=1e-3, 
                       help="黃金標準模型學習率")
    parser.add_argument("--gs_lr_min", type=float, default=1e-6, 
                       help="黃金標準模型最小學習率")
    parser.add_argument("--gs_lr_scheduler", type=str, 
                       choices=['cosine', 'onecycle', 'step'],
                       default='onecycle', help="黃金標準模型學習率調度器")
    parser.add_argument("--gs_weight_decay", type=float, default=0.05, 
                       help="黃金標準模型權重衰減")
    
    # === MIA評估參數 ===
    parser.add_argument("--run_mia", action="store_true", default=True,
                       help="是否運行MIA評估")
    parser.add_argument("--no_mia", action="store_false", dest="run_mia",
                       help="跳過MIA評估")
    parser.add_argument("--mia_epochs", type=int, default=30, 
                       help="MIA攻擊模型訓練輪數")
    parser.add_argument("--mia_lr", type=float, default=1e-3, 
                       help="MIA攻擊模型學習率")
    parser.add_argument("--mia_use_scheduler", action="store_true", default=True,
                       help="MIA是否使用學習率調度器")
    
    # === 增強評估參數 ===
    parser.add_argument("--run_enhanced_eval", action="store_true", default=True,
                       help="是否運行增強評估（cosine similarity等）")
    parser.add_argument("--no_enhanced_eval", action="store_false", dest="run_enhanced_eval",
                       help="跳過增強評估")
    
    # === 結果保存參數 ===
    parser.add_argument("--save_models", action="store_true", default=False,
                       help="是否保存所有中間模型")
    parser.add_argument("--save_detailed_logs", action="store_true", default=True,
                       help="是否保存詳細訓練日誌")
    
    # === 調試參數 ===
    parser.add_argument("--debug", action="store_true", default=False,
                       help="調試模式：使用較少的epochs和數據")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="詳細輸出")
    
    args = parser.parse_args()
    
    # 調試模式調整
    if args.debug:
        args.unlearn_epochs = min(args.unlearn_epochs, 3)
        args.gs_epochs = min(args.gs_epochs, 5)
        args.mia_epochs = min(args.mia_epochs, 3)
        print("🐛 調試模式已啟用，使用較少的訓練輪數")
    
    # 處理遺忘類別配置
    if args.forget_classes is not None:
        # 直接指定的類別優先
        if '-' in args.forget_classes:
            start, end = map(int, args.forget_classes.split('-'))
            forget_classes_configs = [{
                'num_classes': end-start+1, 
                'classes': set(range(start, end+1)), 
                'seed': args.seed
            }]
        else:
            classes = {int(x.strip()) for x in args.forget_classes.split(',') if x.strip()}
            forget_classes_configs = [{
                'num_classes': len(classes), 
                'classes': classes, 
                'seed': args.seed
            }]
    else:
        # 使用動態生成
        forget_classes_configs = []
        for num_classes in args.forget_class_counts:
            forget_classes = generate_forget_classes(num_classes, seed=args.seed)
            forget_classes_configs.append({
                'num_classes': num_classes,
                'classes': forget_classes,
                'seed': args.seed
            })
    
    # 創建實驗器並運行
    experiment = LayerWiseExperiment(args)
    exp_dir, results = experiment.run_all_experiments(forget_classes_configs)
    
    print(f"\n所有LayerWise實驗完成！")
    print(f"結果保存在: {exp_dir}")


if __name__ == "__main__":
    main()