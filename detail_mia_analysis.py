import pandas as pd
import numpy as np
import csv
import os
import torch
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_curve, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def save_mia_analysis_to_csv(mia_results, attack_model, X_test, y_test, original_labels_test, output_dir):
    """
    將MIA攻擊分析結果保存到CSV文件中
    
    Args:
        mia_results: MIA評估結果字典
        attack_model: 訓練好的攻擊模型
        X_test: 測試特徵
        y_test: 測試標籤(1=成員，0=非成員)
        original_labels_test: 原始標籤
        output_dir: 輸出目錄
    """
    # 創建MIA分析目錄
    mia_dir = os.path.join(output_dir, 'mia_analysis')
    os.makedirs(mia_dir, exist_ok=True)
    
    # 1. 獲取模型預測
    attack_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(attack_model.model[0].weight.device)
        y_pred_proba = attack_model(X_test_tensor).cpu().numpy().flatten()
    
    # 2. 創建詳細的預測結果DataFrame
    predictions_df = pd.DataFrame({
        '樣本ID': range(len(y_test)),
        '實際成員狀態': ['成員' if y == 1 else '非成員' for y in y_test],
        '預測概率': y_pred_proba,
        '預測標籤': ['成員' if p > 0.5 else '非成員' for p in y_pred_proba],
        '預測是否正確': [pred == actual for pred, actual in zip(y_pred_proba > 0.5, y_test)],
        '樣本類型': ['保留集' if (y_test[i] == 1) else '遺忘集' if original_labels_test[i] >= 0 else '測試集' 
                    for i in range(len(y_test))]
    })
    
    predictions_df.to_csv(os.path.join(mia_dir, 'detailed_predictions.csv'), index=False, encoding='utf-8')
    
    # 3. 保存ROC曲線數據
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_df = pd.DataFrame({
        '假陽性率': fpr,
        '真陽性率': tpr,
        '閾值': thresholds
    })
    roc_df.to_csv(os.path.join(mia_dir, 'roc_curve_data.csv'), index=False, encoding='utf-8')
    
    # 計算AUC score並保存
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 4. 保存PR曲線數據
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_df = pd.DataFrame({
        '查準率': precision,
        '查全率': recall,
        '閾值': np.append(pr_thresholds, np.nan)  # PR曲線的閾值比點數少1
    })
    pr_df.to_csv(os.path.join(mia_dir, 'precision_recall_curve_data.csv'), index=False, encoding='utf-8')
    
    # 5. 保存不同類型樣本的統計
    sample_stats = []
    
    for sample_type in ['保留集', '遺忘集', '測試集']:
        mask = predictions_df['樣本類型'] == sample_type
        if mask.any():
            type_df = predictions_df[mask]
            stats = {
                '樣本類型': sample_type,
                '總數': len(type_df),
                '預測準確率': type_df['預測是否正確'].mean(),
                '平均預測概率': type_df['預測概率'].mean(),
                '成員預測比例': (type_df['預測標籤'] == '成員').mean()
            }
            sample_stats.append(stats)
    
    sample_stats_df = pd.DataFrame(sample_stats)
    sample_stats_df.to_csv(os.path.join(mia_dir, 'sample_type_statistics.csv'), index=False, encoding='utf-8')
    
    # 6. 保存閾值分析
    thresholds_to_analyze = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_analysis = []
    
    for threshold in thresholds_to_analyze:
        y_pred_binary = (y_pred_proba > threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        
        # 計算不同類型樣本的準確率
        retain_indices = (y_test == 1)
        forget_indices = (y_test == 0) & (original_labels_test >= 0)
        test_indices = (y_test == 0) & (original_labels_test < 0)
        
        retain_acc = accuracy_score(y_test[retain_indices], y_pred_binary[retain_indices]) if np.any(retain_indices) else np.nan
        forget_acc = accuracy_score(y_test[forget_indices], y_pred_binary[forget_indices]) if np.any(forget_indices) else np.nan
        test_acc = accuracy_score(y_test[test_indices], y_pred_binary[test_indices]) if np.any(test_indices) else np.nan
        
        threshold_analysis.append({
            '閾值': threshold,
            '總體準確率': accuracy,
            '保留集準確率': retain_acc,
            '遺忘集準確率': forget_acc,
            '測試集準確率': test_acc
        })
    
    threshold_df = pd.DataFrame(threshold_analysis)
    threshold_df.to_csv(os.path.join(mia_dir, 'threshold_analysis.csv'), index=False, encoding='utf-8')
    
    # 7. 保存分類報告
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    classification_report_str = classification_report(y_test, y_pred_binary, 
                                                     target_names=['非成員', '成員'])
    
    with open(os.path.join(mia_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report_str)
    
    # 8. 保存MIA評估總結，包含AUC score
    summary_data = []
    # 合併mia_results和auc_score
    mia_results_with_auc = mia_results.copy()
    mia_results_with_auc['auc'] = auc_score
    
    for key, value in mia_results_with_auc.items():
        if isinstance(value, float):
            summary_data.append([key, f"{value:.4f}"])
        else:
            summary_data.append([key, str(value)])
    
    with open(os.path.join(mia_dir, 'mia_evaluation_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['指標', '值'])
        writer.writerows(summary_data)
    
    # 9. 保存特徵重要性分析（如果可用）
    if hasattr(attack_model, 'model') and hasattr(attack_model.model[0], 'weight'):
        feature_importance = np.abs(attack_model.model[0].weight.detach().cpu().numpy()[0])
        feature_names = ['置信度', '熵', '方差', '邊際', '峰度']
        
        importance_df = pd.DataFrame({
            '特徵名稱': feature_names,
            '重要性': feature_importance / feature_importance.sum()
        })
        importance_df = importance_df.sort_values('重要性', ascending=False)
        importance_df.to_csv(os.path.join(mia_dir, 'feature_importance.csv'), index=False, encoding='utf-8')
    
    print(f"MIA詳細分析結果已保存至: {mia_dir}")
    return mia_dir


def create_mia_visualization(mia_dir):
    """
    Create visualization charts based on MIA analysis data
    
    Args:
        mia_dir: Directory containing MIA analysis results
    """
    # No need for Chinese font settings anymore
    
    # Read MIA evaluation summary to get AUC value
    try:
        summary_df = pd.read_csv(os.path.join(mia_dir, 'mia_evaluation_summary.csv'))
        # Convert column names to English when reading
        summary_df.columns = ['Metric', 'Value'] 
        auc_dict = dict(zip(summary_df['Metric'], summary_df['Value']))
        auc_score = float(auc_dict.get('auc', '0'))
    except:
        auc_score = 0
    
    # 1. ROC curve
    roc_df = pd.read_csv(os.path.join(mia_dir, 'roc_curve_data.csv'))
    # Convert column names to English
    roc_df.columns = ['False Positive Rate', 'True Positive Rate', 'Threshold']
    
    plt.figure(figsize=(10, 8))
    plt.plot(roc_df['False Positive Rate'], roc_df['True Positive Rate'], 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.text(0.6, 0.2, f'AUC = {auc_score:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(mia_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PR curve
    pr_df = pd.read_csv(os.path.join(mia_dir, 'precision_recall_curve_data.csv'))
    # Convert column names to English
    pr_df.columns = ['Precision', 'Recall', 'Threshold']
    
    plt.figure(figsize=(10, 8))
    plt.plot(pr_df['Recall'], pr_df['Precision'], 'r-', linewidth=2)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(mia_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Attack effectiveness by sample type
    stats_df = pd.read_csv(os.path.join(mia_dir, 'sample_type_statistics.csv'))
    # Convert column names and values to English
    stats_df.columns = ['Sample Type', 'Total', 'Prediction Accuracy', 'Average Prediction Probability', 'Member Prediction Ratio']
    # Map Chinese sample types to English
    sample_type_map = {
        '保留集': 'Retained Set',
        '遺忘集': 'Forgotten Set',
        '測試集': 'Test Set'
    }
    stats_df['Sample Type'] = stats_df['Sample Type'].map(lambda x: sample_type_map.get(x, x))
    
    plt.figure(figsize=(12, 8))
    bar_positions = range(len(stats_df))
    accuracy_bars = plt.bar(bar_positions, stats_df['Prediction Accuracy'], 
                           color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.xticks(bar_positions, stats_df['Sample Type'], fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=14)
    plt.title('MIA Attack Effectiveness by Sample Type', fontsize=16)
    plt.ylim(0, 1.2)
    
    # Add value labels on bars
    for bar in accuracy_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(mia_dir, 'sample_type_attack_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Threshold analysis
    threshold_df = pd.read_csv(os.path.join(mia_dir, 'threshold_analysis.csv'))
    # Convert column names to English
    threshold_df.columns = ['Threshold', 'Overall Accuracy', 'Retained Set Accuracy', 
                           'Forgotten Set Accuracy', 'Test Set Accuracy']
    
    plt.figure(figsize=(14, 10))
    plt.plot(threshold_df['Threshold'], threshold_df['Overall Accuracy'], 
             'o-', label='Overall Accuracy', linewidth=2, markersize=8)
    plt.plot(threshold_df['Threshold'], threshold_df['Retained Set Accuracy'], 
             's-', label='Retained Set Accuracy', linewidth=2, markersize=8)
    plt.plot(threshold_df['Threshold'], threshold_df['Forgotten Set Accuracy'], 
             '^-', label='Forgotten Set Accuracy', linewidth=2, markersize=8)
    plt.plot(threshold_df['Threshold'], threshold_df['Test Set Accuracy'], 
             'd-', label='Test Set Accuracy', linewidth=2, markersize=8)
    
    plt.xlabel('Classification Threshold', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Impact of Threshold on MIA Performance', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(mia_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature importance
    if os.path.exists(os.path.join(mia_dir, 'feature_importance.csv')):
        importance_df = pd.read_csv(os.path.join(mia_dir, 'feature_importance.csv'))
        # Convert column names and values to English
        importance_df.columns = ['Feature Name', 'Importance']
        # Map Chinese feature names to English
        feature_name_map = {
            '置信度': 'Confidence',
            '熵': 'Entropy',
            '方差': 'Variance',
            '邊際': 'Margin',
            '峰度': 'Kurtosis'
        }
        importance_df['Feature Name'] = importance_df['Feature Name'].map(lambda x: feature_name_map.get(x, x))
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(importance_df['Feature Name'], importance_df['Importance'], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.ylabel('Importance Score', fontsize=14)
        plt.title('Feature Importance in MIA Attack', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(mia_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"MIA visualization results saved to: {mia_dir}")