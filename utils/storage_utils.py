"""
儲存工具 - 管理實驗結果的儲存和組織
"""
import os
import json
import csv
import logging
import pandas as pd
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentStorage:
    """實驗儲存管理器 - 統一管理所有實驗的配置、結果和日誌"""
    
    def __init__(self, base_dir="./checkpoints"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def create_complete_experiment_structure(self, method_name, timestamp=None):
        """創建完整的實驗目錄結構"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        exp_name = f"{method_name}_{timestamp}"
        exp_dir = f"{self.base_dir}/{exp_name}"
        
        # 創建完整目錄結構
        directories = [
            exp_dir,
            f"{exp_dir}/results",
            f"{exp_dir}/logs", 
            f"{exp_dir}/configs",
            f"{exp_dir}/models",
            f"{exp_dir}/tensorboard",
            f"{exp_dir}/visualizations"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return exp_dir, exp_name
    
    def setup_logging_system(self, exp_dir, exp_name):
        """設置統一的日誌系統"""
        log_file = f"{exp_dir}/logs/experiment.log"
        
        # 配置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        logger = logging.getLogger(f"{exp_name}_logger")
        
        return logger, log_file
    
    def setup_tensorboard(self, exp_dir):
        """設置TensorBoard目錄結構"""
        tensorboard_dir = f"{exp_dir}/tensorboard"
        main_writer = SummaryWriter(f"{tensorboard_dir}/overview")
        
        return tensorboard_dir, main_writer
    
    def save_complete_experiment_config(self, exp_dir, method_name, args, model_args=None):
        """保存完整的實驗配置"""
        config = {
            'experiment_info': {
                'method': method_name,
                'timestamp': datetime.now().isoformat(),
                'description': f'{method_name}遺忘實驗',
                'experiment_dir': exp_dir
            }
        }
        
        # 添加模型配置
        if model_args:
            config['model_args'] = model_args
        
        # 添加所有參數
        config['all_args'] = vars(args) if hasattr(args, '__dict__') else args
        
        # 根據方法添加特定配置
        if method_name.lower() == 'layerwise':
            config.update({
                'training_args': {
                    'unlearn_epochs': getattr(args, 'unlearn_epochs', 30),
                    'unlearn_lr': getattr(args, 'unlearn_lr', 5e-5),
                    'weight_decay': getattr(args, 'weight_decay', 0.01),
                    'batch_size': getattr(args, 'batch_size', 128)
                }
            })
        elif method_name.lower() in ['weightpruning', 'weight_pruning']:
            config.update({
                'pruning_args': {
                    'unlearn_epochs': getattr(args, 'unlearn_epochs', 30),
                    'unlearn_lr': getattr(args, 'unlearn_lr', 1e-4),
                    'batch_size': getattr(args, 'batch_size', 128)
                }
            })
        elif method_name.lower() in ['attention', 'attentiontargeted']:
            config.update({
                'attention_args': {
                    'unlearn_epochs': getattr(args, 'unlearn_epochs', 30),
                    'unlearn_lr': getattr(args, 'unlearn_lr', 1e-4),
                    'batch_size': getattr(args, 'batch_size', 128)
                }
            })
        
        # 添加黃金標準配置
        config['goldstandard_args'] = {
            'gs_epochs': getattr(args, 'gs_epochs', 250),
            'gs_lr': getattr(args, 'gs_lr', 1e-3),
            'gs_scheduler': getattr(args, 'gs_scheduler', 'onecycle'),
            'gs_weight_decay': getattr(args, 'gs_weight_decay', 0.05)
        }
        
        # 添加評估配置
        config['evaluation_args'] = {
            'run_mia': getattr(args, 'run_mia', True),
            'mia_epochs': getattr(args, 'mia_epochs', 30),
            'run_enhanced_eval': getattr(args, 'run_enhanced_eval', True)
        }
        
        # 保存配置文件
        config_file = f"{exp_dir}/configs/experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config_file, config
    
    def log_experiment_start(self, logger, method_name, exp_name, exp_dir, config, args):
        """統一的實驗開始日誌"""
        logger.info("="*80)
        logger.info(f"{method_name} 遺忘實驗開始")
        logger.info("="*80)
        
        # 基本信息
        logger.info("實驗基本信息:")
        logger.info(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  實驗名稱: {exp_name}")
        logger.info(f"  輸出目錄: {exp_dir}")
        logger.info(f"  方法: {method_name}")
        
        # 記錄重要參數
        if hasattr(args, 'model_path'):
            logger.info(f"  模型路徑: {args.model_path}")
        if hasattr(args, 'device'):
            logger.info(f"  設備: {args.device}")
        
        # 訓練參數
        logger.info("\n訓練參數:")
        for key in ['unlearn_epochs', 'unlearn_lr', 'batch_size', 'weight_decay']:
            if hasattr(args, key):
                logger.info(f"  {key}: {getattr(args, key)}")
        
        # 評估參數
        logger.info("\n評估配置:")
        for key in ['run_mia', 'mia_epochs', 'run_enhanced_eval']:
            if hasattr(args, key):
                logger.info(f"  {key}: {getattr(args, key)}")
        
        logger.info("="*80)
        
        return logger
    
    def save_single_experiment_results(self, exp_dir, method_name, results, strategy_name=None):
        """保存單個實驗結果"""
        # 生成文件名
        if strategy_name:
            filename = f"{method_name}_{strategy_name}_results.json"
        else:
            filename = f"{method_name}_results.json"
        
        # 保存JSON結果
        json_path = self.save_results_to_json(exp_dir, results, filename)
        
        # 保存CSV格式（便於分析）
        csv_filename = filename.replace('.json', '.csv')
        csv_path = self.save_results_to_csv(exp_dir, results, csv_filename)
        
        return json_path, csv_path
    
    def generate_comprehensive_summary(self, exp_dir, method_name, all_results):
        """生成綜合摘要報告"""
        # 生成CSV摘要
        summary_data = []
        for experiment_key, strategies in all_results.items():
            if isinstance(strategies, dict):
                for strategy_name, results in strategies.items():
                    if 'error' not in results:
                        row = self._extract_summary_row(experiment_key, strategy_name, results, method_name)
                        summary_data.append(row)
            else:
                # 單個結果
                if 'error' not in strategies:
                    row = self._extract_summary_row(experiment_key, method_name, strategies, method_name)
                    summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv = f"{exp_dir}/results/{method_name}_comprehensive_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
            
            # 生成文字報告
            report_path = self._generate_text_report(exp_dir, method_name, all_results, summary_df)
            
            return summary_csv, report_path
        else:
            return None, None
    
    def _extract_summary_row(self, experiment_key, strategy_name, results, method_name):
        """提取摘要行數據"""
        row = {
            'method': method_name,
            'experiment': experiment_key,
            'strategy': strategy_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # 提取關鍵指標
        if 'performance_metrics' in results:
            perf = results['performance_metrics']
            row.update({
                'original_retain_acc': perf.get('original_retain_acc', 0),
                'original_forget_acc': perf.get('original_forget_acc', 0),
                'unlearned_retain_acc': perf.get('unlearned_retain_acc', 0),
                'unlearned_forget_acc': perf.get('unlearned_forget_acc', 0),
                'gs_retain_acc': perf.get('gs_retain_acc', 0),
                'gs_forget_acc': perf.get('gs_forget_acc', 0)
            })
        
        if 'unlearning_metrics' in results:
            unlearn = results['unlearning_metrics']
            row.update({
                'zrf_score': unlearn.get('zrf_score', 0),
                'kl_divergence': unlearn.get('kl_divergence', 0),
                'forget_effectiveness': unlearn.get('forget_effectiveness', 0),
                'retain_preservation': unlearn.get('retain_preservation', 0)
            })
        
        if 'timing' in results:
            timing = results['timing']
            row.update({
                'unlearning_time': timing.get('unlearning_time', 0),
                'total_experiment_time': timing.get('total_experiment_time', 0)
            })
        
        return row
    
    def _generate_text_report(self, exp_dir, method_name, all_results, summary_df):
        """生成詳細文字報告"""
        report_path = f"{exp_dir}/results/{method_name}_detailed_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{method_name} 遺忘實驗詳細報告\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 實驗概述
            total_experiments = len(summary_df)
            successful_experiments = len(summary_df[summary_df['zrf_score'] > 0])
            
            f.write(f"實驗概述:\n")
            f.write(f"  總實驗數: {total_experiments}\n")
            f.write(f"  成功實驗: {successful_experiments}\n")
            f.write(f"  成功率: {successful_experiments/max(total_experiments,1)*100:.1f}%\n\n")
            
            # 最佳結果
            if not summary_df.empty and 'zrf_score' in summary_df.columns:
                best_result = summary_df.loc[summary_df['zrf_score'].idxmax()]
                f.write(f"最佳結果:\n")
                f.write(f"  實驗: {best_result['experiment']}\n")
                f.write(f"  策略: {best_result['strategy']}\n")
                f.write(f"  ZRF分數: {best_result['zrf_score']:.6f}\n")
                if 'unlearned_retain_acc' in best_result:
                    f.write(f"  保留準確率: {best_result['unlearned_retain_acc']:.2f}%\n")
                f.write("\n")
            
            # 詳細統計
            if 'zrf_score' in summary_df.columns:
                f.write(f"ZRF分數統計:\n")
                f.write(f"  平均: {summary_df['zrf_score'].mean():.6f}\n")
                f.write(f"  標準差: {summary_df['zrf_score'].std():.6f}\n")
                f.write(f"  最大值: {summary_df['zrf_score'].max():.6f}\n")
                f.write(f"  最小值: {summary_df['zrf_score'].min():.6f}\n\n")
            
            f.write("="*80 + "\n")
        
        return report_path
    
    def create_experiment_dir(self, method_name, timestamp=None):
        """創建實驗目錄"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        
        exp_dir = f"{self.base_dir}/{method_name}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # 創建子目錄
        subdirs = ['models', 'results', 'visualizations', 'logs']
        for subdir in subdirs:
            os.makedirs(f"{exp_dir}/{subdir}", exist_ok=True)
        
        return exp_dir
    
    def save_experiment_config(self, exp_dir, config):
        """保存實驗配置"""
        config_path = f"{exp_dir}/experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return config_path
    
    def save_model(self, exp_dir, model, model_name):
        """保存模型"""
        model_path = f"{exp_dir}/models/{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        return model_path
    
    def save_results_to_csv(self, exp_dir, results, filename):
        """保存結果到CSV"""
        csv_path = f"{exp_dir}/results/{filename}"
        
        # 將嵌套字典展平
        flattened_results = self._flatten_dict(results)
        
        # 創建DataFrame
        df = pd.DataFrame([flattened_results])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path
    
    def save_results_to_json(self, exp_dir, results, filename):
        """保存結果到JSON"""
        json_path = f"{exp_dir}/results/{filename}"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return json_path
    
    def save_detailed_report(self, exp_dir, report_content, filename):
        """保存詳細報告"""
        report_path = f"{exp_dir}/results/{filename}"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return report_path
    
    def append_to_summary_csv(self, summary_path, results):
        """追加結果到總結CSV"""
        flattened_results = self._flatten_dict(results)
        
        # 檢查文件是否存在
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df = pd.concat([df, pd.DataFrame([flattened_results])], ignore_index=True)
        else:
            df = pd.DataFrame([flattened_results])
        
        df.to_csv(summary_path, index=False, encoding='utf-8')
        return summary_path
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def save_comparative_analysis(base_dir, all_results, timestamp=None):
    """保存比較分析結果"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    analysis_dir = f"{base_dir}/comparative_analysis_{timestamp}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 創建總結表格
    summary_data = []
    for method_name, method_results in all_results.items():
        for forget_classes, results in method_results.items():
            row = {
                'method': method_name,
                'forget_classes': forget_classes,
                'timestamp': timestamp
            }
            
            # 基本性能指標
            if 'unlearned_metrics' in results:
                row['retain_acc'] = results['unlearned_metrics'].get('retain_acc', 0)
                row['forget_acc'] = results['unlearned_metrics'].get('forget_acc', 0)
            
            if 'retrained_metrics' in results:
                row['gs_retain_acc'] = results['retrained_metrics'].get('retain_acc', 0)
            
            # 遺忘指標
            row['zrf_score'] = results.get('zrf_score', 0)
            row['kl_divergence'] = results.get('kl_divergence', 0)
            
            # 新指標
            if 'enhanced_metrics' in results:
                enhanced = results['enhanced_metrics']
                row['cosine_similarity'] = enhanced.get('cosine_similarity', 0)
                row['activation_distance'] = enhanced.get('activation_distance', 0)
                
                stat_dist = enhanced.get('statistical_distances', {})
                row['total_variation'] = stat_dist.get('total_variation', 0)
                row['js_divergence'] = stat_dist.get('js_divergence', 0)
                
                feat_attr = enhanced.get('feature_attribution', {})
                row['attribution_ratio'] = feat_attr.get('attribution_ratio', 0)
            
            # 時間指標
            row['unlearning_time'] = results.get('unlearning_time', 0)
            row['retraining_time'] = results.get('retraining_time', 0)
            
            summary_data.append(row)
    
    # 保存總結CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = f"{analysis_dir}/comparative_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    
    # 保存詳細JSON
    detailed_json_path = f"{analysis_dir}/detailed_results.json"
    with open(detailed_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    return analysis_dir, summary_csv_path, detailed_json_path


def generate_method_comparison_report(all_results, output_path):
    """生成方法比較報告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("機器遺忘方法比較分析報告\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 方法概述
        f.write("實驗方法概述:\n")
        f.write("-"*50 + "\n")
        for method_name in all_results.keys():
            f.write(f"• {method_name}\n")
        f.write("\n")
        
        # 每種遺忘類別數的比較
        forget_class_nums = set()
        for method_results in all_results.values():
            forget_class_nums.update(method_results.keys())
        
        for forget_num in sorted(forget_class_nums):
            f.write(f"遺忘 {forget_num} 個類別的結果比較:\n")
            f.write("-"*50 + "\n")
            
            method_comparison = []
            for method_name, method_results in all_results.items():
                if forget_num in method_results:
                    results = method_results[forget_num]
                    
                    retain_acc = 0
                    if 'unlearned_metrics' in results:
                        retain_acc = results['unlearned_metrics'].get('retain_acc', 0)
                    
                    zrf_score = results.get('zrf_score', 0)
                    time_cost = results.get('unlearning_time', 0)
                    
                    method_comparison.append({
                        'method': method_name,
                        'retain_acc': retain_acc,
                        'zrf_score': zrf_score,
                        'time': time_cost
                    })
            
            # 排序並輸出
            method_comparison.sort(key=lambda x: x['zrf_score'], reverse=True)
            
            for i, result in enumerate(method_comparison, 1):
                f.write(f"{i}. {result['method']}:\n")
                f.write(f"   保留準確率: {result['retain_acc']:.2f}%\n")
                f.write(f"   ZRF分數: {result['zrf_score']:.4f}\n")
                f.write(f"   訓練時間: {result['time']:.1f}秒\n\n")
        
        # 總體排名
        f.write("總體方法排名 (基於ZRF分數):\n")
        f.write("-"*50 + "\n")
        
        overall_scores = {}
        for method_name, method_results in all_results.items():
            scores = []
            for results in method_results.values():
                scores.append(results.get('zrf_score', 0))
            overall_scores[method_name] = sum(scores) / len(scores) if scores else 0
        
        sorted_methods = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (method, avg_score) in enumerate(sorted_methods, 1):
            f.write(f"{i}. {method}: 平均ZRF分數 {avg_score:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    return output_path
