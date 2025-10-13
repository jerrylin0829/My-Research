"""
MIA 特徵分布視覺化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
warnings.filterwarnings('ignore')

# 設置 matplotlib 不顯示圖形
plt.ioff()
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_feature_distributions(retain_features, forget_features, test_features, 
                             feature_names=None, save_path=None):
    """
    繪製各組特徵分布對比圖
    """
    if feature_names is None:
        feature_names = ['confidence', 'entropy', 'margin', 'loss']
    
    n_features = retain_features.shape[1]
    feature_names = feature_names[:n_features]

    # 🛠️ 動態計算最佳布局
    if n_features <= 2:
        nrows, ncols = 1, n_features
    elif n_features <= 4:
        nrows, ncols = 2, 2  # 2x2 布局
    elif n_features <= 6:
        nrows, ncols = 2, 3  # 2x3 布局
    elif n_features <= 8:
        nrows, ncols = 2, 4  # 2x4 布局
    else:
        nrows, ncols = 3, (n_features + 2) // 3  # 3行布局

    # 🛠️ 創建適當數量的子圖
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    
    # 確保 axes 是數組格式
    if n_features == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # 繪製分布
        ax.hist(retain_features[:, i], bins=50, alpha=0.6, density=True, 
               color='blue', label=f'Retain (Member)\n({len(retain_features):,} samples)')
        ax.hist(forget_features[:, i], bins=50, alpha=0.6, density=True, 
               color='red', label=f'Forget (Non-member)\n({len(forget_features):,} samples)')
        ax.hist(test_features[:, i], bins=30, alpha=0.4, density=True, 
               color='green', label=f'Test (Non-member)\n({len(test_features):,} samples)')
        
        # 計算統計量
        retain_mean = retain_features[:, i].mean()
        forget_mean = forget_features[:, i].mean()
        test_mean = test_features[:, i].mean()
        
        # 添加均值線
        ax.axvline(retain_mean, color='blue', linestyle='--', linewidth=2)
        ax.axvline(forget_mean, color='red', linestyle='--', linewidth=2)
        ax.axvline(test_mean, color='green', linestyle='--', linewidth=2)
        
        # 計算重疊度
        overlap_retain_forget = calculate_overlap(retain_features[:, i], forget_features[:, i])
        overlap_retain_test = calculate_overlap(retain_features[:, i], test_features[:, i])
        
        ax.set_title(f'{feature_names[i]}\n'
                    f'R-F overlap: {overlap_retain_forget:.2f}, '
                    f'R-T overlap: {overlap_retain_test:.2f}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 特徵分布圖已保存: {save_path}")
    plt.close()
    
    # 打印統計摘要
    print_feature_statistics(retain_features, forget_features, test_features, feature_names)


def calculate_overlap(dist1, dist2):
    """計算兩個分布的重疊度"""
    # 使用直方圖估計重疊
    bins = np.linspace(min(min(dist1), min(dist2)), max(max(dist1), max(dist2)), 50)
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)
    
    # 計算重疊面積
    overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
    return overlap


def plot_2d_feature_scatter(retain_features, forget_features, test_features, 
                           feature_names=None, save_path=None):
    """
    繪製關鍵特徵的2D散點圖
    """
    if feature_names is None:
        feature_names = ['confidence', 'entropy', 'variance', 'margin']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 關鍵特徵對組合
    feature_pairs = [
        (0, 1),  # confidence vs entropy
        (0, 2),  # confidence vs variance  
        (0, 3),  # confidence vs margin
        (1, 2),  # entropy vs variance
        (1, 3),  # entropy vs margin
        (2, 3),  # variance vs margin
    ]
    
    for idx, (i, j) in enumerate(feature_pairs):
        ax = axes[idx]
        
        # 隨機採樣以避免過度擁擠
        n_samples = min(2000, len(retain_features), len(forget_features))
        
        retain_idx = np.random.choice(len(retain_features), n_samples, False)
        forget_idx = np.random.choice(len(forget_features), min(n_samples, len(forget_features)), False)
        test_idx = np.random.choice(len(test_features), min(n_samples, len(test_features)), False)
        
        # 繪製散點圖
        ax.scatter(retain_features[retain_idx, i], retain_features[retain_idx, j], 
                  alpha=0.5, s=20, color='blue', label='Retain (Member)')
        ax.scatter(forget_features[forget_idx, i], forget_features[forget_idx, j], 
                  alpha=0.5, s=20, color='red', label='Forget (Non-member)')
        ax.scatter(test_features[test_idx, i], test_features[test_idx, j], 
                  alpha=0.3, s=10, color='green', label='Test (Non-member)')
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(f'{feature_names[i]} vs {feature_names[j]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 2D散點圖已保存: {save_path}")
    plt.close()


def plot_pca_visualization(retain_features, forget_features, test_features, save_path=None):
    """
    使用 PCA 進行降維視覺化
    """
    # 合併數據
    all_features = np.vstack([retain_features, forget_features, test_features])
    labels = np.concatenate([
        np.ones(len(retain_features)) * 0,      # Retain
        np.ones(len(forget_features)) * 1,      # Forget  
        np.ones(len(test_features)) * 2         # Test
    ])
    
    # PCA 降維
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    # 分組
    retain_2d = features_2d[labels == 0]
    forget_2d = features_2d[labels == 1] 
    test_2d = features_2d[labels == 2]
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA 散點圖
    ax1.scatter(retain_2d[:, 0], retain_2d[:, 1], alpha=0.6, s=20, 
               color='blue', label=f'Retain (Member) - {len(retain_2d)}')
    ax1.scatter(forget_2d[:, 0], forget_2d[:, 1], alpha=0.6, s=20, 
               color='red', label=f'Forget (Non-member) - {len(forget_2d)}')
    ax1.scatter(test_2d[:, 0], test_2d[:, 1], alpha=0.4, s=10, 
               color='green', label=f'Test (Non-member) - {len(test_2d)}')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('PCA Visualization of MIA Features')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 特徵重要性
    feature_names = ['confidence', 'entropy', 'margin', 'loss']
    
    pc1_importance = abs(pca.components_[0])
    pc2_importance = abs(pca.components_[1])
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    ax2.bar(x - width/2, pc1_importance, width, label='PC1', alpha=0.8)
    ax2.bar(x + width/2, pc2_importance, width, label='PC2', alpha=0.8)
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Absolute Loading')
    ax2.set_title('PCA Feature Importance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ PCA視覺化已保存: {save_path}")
    plt.close()
    
    print(f"📊 PCA 解釋方差: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
          f"總計={pca.explained_variance_ratio_[:2].sum():.1%}")


def plot_tsne_visualization(retain_features, forget_features, test_features, save_path=None):
    """
    使用 t-SNE 進行非線性降維視覺化
    """
    # 限制樣本數量以加速 t-SNE
    max_samples = 3000
    
    retain_subset = retain_features[:min(max_samples, len(retain_features))]
    forget_subset = forget_features[:min(max_samples, len(forget_features))]
    test_subset = test_features[:min(max_samples, len(test_features))]
    
    # 合併數據
    all_features = np.vstack([retain_subset, forget_subset, test_subset])
    labels = np.concatenate([
        np.ones(len(retain_subset)) * 0,
        np.ones(len(forget_subset)) * 1,
        np.ones(len(test_subset)) * 2
    ])
    
    # t-SNE 降維
    print("執行 t-SNE 降維（可能需要幾分鐘）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)
    
    # 分組
    retain_2d = features_2d[labels == 0]
    forget_2d = features_2d[labels == 1]
    test_2d = features_2d[labels == 2]
    
    # 繪圖
    plt.figure(figsize=(12, 8))
    
    plt.scatter(retain_2d[:, 0], retain_2d[:, 1], alpha=0.6, s=20, 
               color='blue', label=f'Retain (Member) - {len(retain_2d)}')
    plt.scatter(forget_2d[:, 0], forget_2d[:, 1], alpha=0.6, s=20, 
               color='red', label=f'Forget (Non-member) - {len(forget_2d)}')
    plt.scatter(test_2d[:, 0], test_2d[:, 1], alpha=0.4, s=10, 
               color='green', label=f'Test (Non-member) - {len(test_2d)}')
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of MIA Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE視覺化已保存: {save_path}")
    plt.close()


def create_summary_report(retain_features, forget_features, test_features, output_dir):
    """
    創建文字摘要報告
    """
    feature_names = ['confidence', 'entropy', 'margin', 'loss']
    
    report_path = os.path.join(output_dir, "feature_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MIA 特徵分析報告\n")
        f.write("="*80 + "\n\n")
        
        # 基本統計
        f.write("📊 數據量統計:\n")
        f.write(f"保留集 (成員): {len(retain_features):,} 樣本\n")
        f.write(f"遺忘集 (非成員): {len(forget_features):,} 樣本\n")
        f.write(f"測試集 (非成員): {len(test_features):,} 樣本\n\n")
        
        # 特徵統計
        f.write("📈 特徵統計:\n")
        f.write(f"{'特徵名稱':<12} {'保留均值':<10} {'遺忘均值':<10} {'測試均值':<10} {'R-F差異':<10} {'重疊度':<10}\n")
        f.write("-" * 80 + "\n")
        
        feature_stats = []
        for i, name in enumerate(feature_names):
            retain_vals = retain_features[:, i]
            forget_vals = forget_features[:, i]
            test_vals = test_features[:, i]
            
            retain_mean = retain_vals.mean()
            forget_mean = forget_vals.mean()
            test_mean = test_vals.mean()
            diff = abs(retain_mean - forget_mean)
            overlap = calculate_overlap(retain_vals, forget_vals)
            
            f.write(f"{name:<12} {retain_mean:<10.4f} {forget_mean:<10.4f} "
                   f"{test_mean:<10.4f} {diff:<10.4f} {overlap:<10.3f}\n")
            
            feature_stats.append({
                'name': name,
                'diff': diff,
                'overlap': overlap,
                'retain_mean': retain_mean,
                'forget_mean': forget_mean
            })
        
        # 分析結論
        f.write(f"\n🎯 關鍵發現:\n")
        
        # 最具區分性的特徵
        feature_stats.sort(key=lambda x: x['diff'], reverse=True)
        f.write(f"最具區分性的特徵:\n")
        for i, stat in enumerate(feature_stats[:3]):
            f.write(f"  {i+1}. {stat['name']}: 差異={stat['diff']:.4f}\n")
        
        # 重疊度最高的特徵
        feature_stats.sort(key=lambda x: x['overlap'], reverse=True)
        f.write(f"\n重疊度最高的特徵:\n")
        for i, stat in enumerate(feature_stats[:3]):
            f.write(f"  {i+1}. {stat['name']}: 重疊度={stat['overlap']:.3f}\n")
        
        # 問題診斷
        f.write(f"\n🚨 問題診斷:\n")
        
        # 檢查過度自信
        confidence_high_retain = (retain_features[:, 0] > 0.95).mean()
        confidence_high_forget = (forget_features[:, 0] > 0.95).mean()
        
        if confidence_high_retain > 0.8:
            f.write(f"⚠️ 保留集過度自信: {confidence_high_retain:.1%} 樣本置信度 >0.95\n")
        
        if confidence_high_forget > 0.5:
            f.write(f"⚠️ 遺忘集仍然自信: {confidence_high_forget:.1%} 樣本置信度 >0.95\n")
            f.write("  → 表明遺忘效果不足\n")
        
        # 檢查特徵變異性
        low_variance_features = []
        for i, name in enumerate(feature_names):
            if retain_features[:, i].var() < 0.01:
                low_variance_features.append(name)
        
        if low_variance_features:
            f.write(f"⚠️ 低變異性特徵: {', '.join(low_variance_features)}\n")
            f.write("  → 特徵缺乏區分能力\n")
        
        f.write(f"\n💡 建議:\n")
        if confidence_high_forget > 0.7:
            f.write("1. 增加遺忘強度 (提高修剪比例到30-50%)\n")
        if len(low_variance_features) > 3:
            f.write("2. 增加更敏感的特徵 (如loss-based特徵)\n")
        if all(stat['overlap'] > 0.8 for stat in feature_stats[:4]):
            f.write("3. 考慮使用更激進的遺忘方法\n")
    
    print(f"✅ 分析報告已保存: {report_path}")


def print_feature_statistics(retain_features, forget_features, test_features, feature_names):
    """打印詳細的特徵統計"""
    print("\n" + "="*80)
    print("📊 特徵統計分析")
    print("="*80)
    
    feature_stats = []
    
    for i, name in enumerate(feature_names):
        retain_vals = retain_features[:, i]
        forget_vals = forget_features[:, i]
        test_vals = test_features[:, i]
        
        stats = {
            'feature': name,
            'retain_mean': retain_vals.mean(),
            'forget_mean': forget_vals.mean(),
            'test_mean': test_vals.mean(),
            'retain_std': retain_vals.std(),
            'forget_std': forget_vals.std(),
            'test_std': test_vals.std(),
            'retain_forget_diff': abs(retain_vals.mean() - forget_vals.mean()),
            'retain_test_diff': abs(retain_vals.mean() - test_vals.mean()),
            'overlap_rf': calculate_overlap(retain_vals, forget_vals),
            'overlap_rt': calculate_overlap(retain_vals, test_vals)
        }
        feature_stats.append(stats)
    
    # 創建 DataFrame 便於查看
    df = pd.DataFrame(feature_stats)
    
    print("📈 各特徵統計摘要:")
    print(f"{'特徵':<12} {'保留均值':<8} {'遺忘均值':<8} {'測試均值':<8} {'R-F差異':<8} {'R-F重疊':<8}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['feature']:<12} {row['retain_mean']:<8.3f} {row['forget_mean']:<8.3f} "
              f"{row['test_mean']:<8.3f} {row['retain_forget_diff']:<8.3f} {row['overlap_rf']:<8.3f}")
    
    # 找出最具區分性的特徵
    df_sorted = df.sort_values('retain_forget_diff', ascending=False)
    print(f"\n🎯 最具區分性的特徵:")
    for i, (_, row) in enumerate(df_sorted.head(3).iterrows()):
        print(f"{i+1}. {row['feature']}: 差異={row['retain_forget_diff']:.4f}, "
              f"重疊度={row['overlap_rf']:.3f}")
    
    print(f"\n⚠️  重疊度最高的特徵:")
    df_overlap = df.sort_values('overlap_rf', ascending=False)
    for i, (_, row) in enumerate(df_overlap.head(3).iterrows()):
        print(f"{i+1}. {row['feature']}: 重疊度={row['overlap_rf']:.3f}, "
              f"差異={row['retain_forget_diff']:.4f}")


def analyze_feature_overlap(retain_features, forget_features, test_features):
    """分析特徵重疊的原因"""
    print("\n" + "="*80)
    print("🔍 特徵重疊原因分析")
    print("="*80)
    
    feature_names = ['confidence', 'entropy', 'margin', 'loss']
    
    # 1. 檢查極值分布
    print("1️⃣ 極值分布檢查:")
    for i, name in enumerate(feature_names):
        retain_extreme = np.sum((retain_features[:, i] > 0.95) | (retain_features[:, i] < 0.05))
        forget_extreme = np.sum((forget_features[:, i] > 0.95) | (forget_features[:, i] < 0.05))
        
        retain_ratio = retain_extreme / len(retain_features)
        forget_ratio = forget_extreme / len(forget_features)
        
        if retain_ratio > 0.8 or forget_ratio > 0.8:
            print(f"   ⚠️ {name}: 保留集 {retain_ratio:.1%}, 遺忘集 {forget_ratio:.1%} 為極值")
    
    # 2. 檢查方差
    print("\n2️⃣ 特徵方差分析:")
    for i, name in enumerate(feature_names):
        retain_var = retain_features[:, i].var()
        forget_var = forget_features[:, i].var()
        
        if retain_var < 0.01 and forget_var < 0.01:
            print(f"   ⚠️ {name}: 方差過小 (保留:{retain_var:.4f}, 遺忘:{forget_var:.4f})")
    
    # 3. 檢查是否所有樣本都過度自信
    confidence_high = (retain_features[:, 0] > 0.95).mean()
    entropy_low = (retain_features[:, 1] < 0.1).mean()
    
    print(f"\n3️⃣ 模型自信度檢查:")
    print(f"   保留集高置信度 (>0.95): {confidence_high:.1%}")
    print(f"   保留集低熵 (<0.1): {entropy_low:.1%}")
    
    if confidence_high > 0.8:
        print("   🚨 模型對保留集過度自信，可能導致特徵區分度不足")
    
    forget_confidence_high = (forget_features[:, 0] > 0.95).mean()
    forget_entropy_low = (forget_features[:, 1] < 0.1).mean()
    
    print(f"   遺忘集高置信度 (>0.95): {forget_confidence_high:.1%}")
    print(f"   遺忘集低熵 (<0.1): {forget_entropy_low:.1%}")
    
    if forget_confidence_high > 0.5:
        print("   🚨 模型對遺忘集仍然過度自信，遺忘效果不足")


# 主要調用函數
def visualize_mia_features(retain_features, forget_features, test_features, 
                          output_dir="./mia_analysis"):
    """
    完整的 MIA 特徵視覺化分析
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # print("🎨 開始 MIA 特徵視覺化分析...")
    print(f"[Debug] retain={retain_features.shape}, forget={forget_features.shape}, test={test_features.shape}")
    
    # 1. 特徵分布圖
    print("\n1️⃣ 繪製特徵分布圖...")
    plot_feature_distributions(
        retain_features, forget_features, test_features,
        save_path=os.path.join(output_dir, "feature_distributions.png")
    )
    
    # 2. 2D 散點圖
    print("\n2️⃣ 繪製2D特徵散點圖...")
    plot_2d_feature_scatter(
        retain_features, forget_features, test_features,
        save_path=os.path.join(output_dir, "feature_scatter_2d.png")
    )
    
    # 3. PCA 視覺化
    print("\n3️⃣ PCA 降維視覺化...")
    plot_pca_visualization(
        retain_features, forget_features, test_features,
        save_path=os.path.join(output_dir, "pca_visualization.png")
    )
    
    # 4. t-SNE 視覺化
    print("\n4️⃣ t-SNE 非線性降維視覺化...")
    plot_tsne_visualization(
        retain_features, forget_features, test_features,
        save_path=os.path.join(output_dir, "tsne_visualization.png")
    )
    
    # 5. 重疊原因分析
    print("\n5️⃣ 分析特徵重疊原因...")
    analyze_feature_overlap(retain_features, forget_features, test_features)
    
    # 6. 創建文字報告
    print("\n6️⃣ 生成分析報告...")
    create_summary_report(retain_features, forget_features, test_features, output_dir)
    
    print(f"\n✅ 視覺化完成！所有文件保存在: {output_dir}")
    print(f"📁 生成的文件:")
    print(f"   - feature_distributions.png: 特徵分布對比")
    print(f"   - feature_scatter_2d.png: 2D特徵散點圖")
    print(f"   - pca_visualization.png: PCA降維視覺化")
    print(f"   - tsne_visualization.png: t-SNE非線性降維")
    print(f"   - feature_analysis_report.txt: 詳細分析報告")