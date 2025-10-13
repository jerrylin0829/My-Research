import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

def calc_kldscore(model_a: nn.Module, model_b: nn.Module, dataloader: DataLoader, device="cuda", eps: float = 1e-10) -> float:
    """
    對稱 KL 散度（平均 KL(p||q) 與 KL(q||p)），對整個 dataloader 取 batchmean。
    回傳單一 float，數值越小代表兩模型輸出分佈越接近。
    """
    model_a.eval()
    model_b.eval()
    kl_vals = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            pa = F.softmax(model_a(x), dim=1).clamp(min=eps)
            pb = F.softmax(model_b(x), dim=1).clamp(min=eps)

            # KL(p||q) = sum p * (log p - log q)
            kl_ab = (pa * (pa.log() - pb.log())).sum(dim=1).mean()
            kl_ba = (pb * (pb.log() - pa.log())).sum(dim=1).mean()
            kl_sym = 0.5 * (kl_ab + kl_ba)
            kl_vals.append(kl_sym.item())

    return float(np.mean(kl_vals)) if kl_vals else 0.0

def JSDiv(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10, normalize: bool = True) -> torch.Tensor:
    """
    Jensen–Shannon Divergence（批次平均）
      - normalize=True: JS ∈ [0, 1]
      - normalize=False: JS ∈ [0, ln 2]
    p, q: [N, C] 機率分佈（softmax 後）
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)

    # JS = 0.5 KL(p||m) + 0.5 KL(q||m)
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
    js = 0.5 * (kl_pm + kl_qm)

    if normalize:
        js = js / np.log(2)  # 映射到 [0, 1]
    return js

def calc_zrf(unlearned_model: nn.Module,
             random_model: nn.Module,
             forget_loader: DataLoader,
             device: str = "cuda") -> float:
    """
    Zero Retrain Forgetting (ZRF)
    與「隨機初始化模型」比較遺忘模型在 forget set 上的輸出分佈相似度：
      ZRF = 1 - JS( p_{M_f} || p_{M_rand} )
    值越接近 1 代表越像「完全沒學過」→ 遺忘越徹底
    """
    unlearned_model.eval()
    random_model.eval()

    preds_u, preds_r = [], []
    with torch.no_grad():
        for batch in forget_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            preds_u.append(F.softmax(unlearned_model(x), dim=1).cpu())
            preds_r.append(F.softmax(random_model(x), dim=1).cpu())

    if not preds_u:
        return 0.0

    preds_u = torch.cat(preds_u, dim=0)
    preds_r = torch.cat(preds_r, dim=0)
    js = JSDiv(preds_u, preds_r, normalize=True)
    zrf = float(torch.clamp(1.0 - js, 0.0, 1.0))
    return zrf

def alignment_score(unlearned_model: nn.Module,
                    gold_model: nn.Module,
                    forget_loader: DataLoader,
                    device: str = "cuda",
                    normalize_js: bool = True) -> float:
    """
    與 gold/retrain 模型在 forget set 上的對齊度（越高越好）：
      Align = 1 - JS( p_{M_f} || p_{M_gold} )
    這個指標衡量「遺忘模型是否接近理想的重訓模型」，常被稱作 AG/Alignment Gap 的反向分數。
    """
    unlearned_model.eval()
    gold_model.eval()

    preds_u, preds_g = [], []
    with torch.no_grad():
        for batch in forget_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            preds_u.append(F.softmax(unlearned_model(x), dim=1).cpu())
            preds_g.append(F.softmax(gold_model(x), dim=1).cpu())

    if not preds_u:
        return 0.0

    preds_u = torch.cat(preds_u, dim=0)
    preds_g = torch.cat(preds_g, dim=0)
    js = JSDiv(preds_u, preds_g, normalize=normalize_js)
    align = float(torch.clamp(1.0 - js, 0.0, 1.0))
    return align

def calc_kldscore(model_a: nn.Module, model_b: nn.Module, dataloader: DataLoader,
                  device: str = "cuda", eps: float = 1e-10) -> float:
    """
    對稱 KL（平均 KL(p||q) 與 KL(q||p)）—可做輔助參考
    """
    model_a.eval()
    model_b.eval()
    kl_vals = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            pa = F.softmax(model_a(x), dim=1).clamp(min=eps)
            pb = F.softmax(model_b(x), dim=1).clamp(min=eps)

            kl_ab = (pa * (pa.log() - pb.log())).sum(dim=1).mean()
            kl_ba = (pb * (pb.log() - pa.log())).sum(dim=1).mean()
            kl_sym = 0.5 * (kl_ab + kl_ba)
            kl_vals.append(kl_sym.item())

    return float(np.mean(kl_vals)) if kl_vals else 0.0

def entropy(p, dim=-1, keepdim=False, eps: float = 1e-12):
    """
    計算熵
    H(p) = -sum p log p
    使用 clamp 避免 log(0)

    Args:
        p: 概率分佈
        dim: 計算維度
        keepdim: 是否保持維度
    
    Returns:
        熵值
    """
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    """
    收集模型對數據集的預測概率
    
    Args:
        data_loader: 數據加載器
        model: 模型
    
    Returns:
        預測概率張量
    """
    # 創建單樣本數據加載器以獲得更準確的概率估計
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,  # 調低worker數以減少內存壓力
        pin_memory=True
    )
    
    prob = []
    model.eval()
    
    dev = next(model.parameters()).device

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = [tensor.to(dev) for tensor in batch if torch.is_tensor(tensor)]
                data = batch[0]
            else:
                data = batch.to(dev)
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data.cpu())

    return torch.cat(prob)

def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    """
    準備成員推斷攻擊所需的數據
    
    Args:
        retain_loader: 保留數據加載器
        forget_loader: 遺忘數據加載器
        test_loader: 測試數據加載器
        model: 模型
    
    Returns:
        X_f, Y_f, X_r, Y_r: 攻擊特徵和標籤
    """
    # 收集概率預測
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)
    
    # 準備訓練數據（保留集+測試集）
    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])
    
    # 準備遺忘集數據
    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.ones(len(forget_prob))
    
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    """
    執行成員推斷攻擊並返回攻擊準確率
    
    Args:
        retain_loader: 保留數據加載器
        forget_loader: 遺忘數據加載器
        test_loader: 測試數據加載器
        model: 模型
    
    Returns:
        遺忘集上的攻擊準確率（應該盡可能低）
    """
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    
    # 使用SVM作為攻擊模型
    clf = SVC(C=3, gamma='auto', kernel='rbf', probability=True)
    # 可選：使用邏輯回歸
    # clf = LogisticRegression(class_weight='balanced', solver='lbfgs', multi_class='multinomial')
    
    # 訓練攻擊模型
    clf.fit(X_r, Y_r)
    
    # 預測遺忘集
    results = clf.predict(X_f)
    
    # 返回被識別為成員的比例（應該盡可能低）
    return float(results.mean())


def get_membership_attack_scores(retain_loader, forget_loader, test_loader, model):
    """
    獲取詳細的成員推斷攻擊分數
    
    Args:
        retain_loader: 保留數據加載器
        forget_loader: 遺忘數據加載器
        test_loader: 測試數據加載器
        model: 模型
    
    Returns:
        包含各種MIA指標的字典
    """
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    
    # 訓練攻擊模型
    clf = SVC(C=3, gamma='auto', kernel='rbf', probability=True)
    clf.fit(X_r, Y_r)
    
    # 獲取預測概率
    forget_probs = clf.predict_proba(X_f)[:, 1]  # 概率為成員的可能性
    
    # 計算各種指標
    forget_predictions = clf.predict(X_f)
    
    # 準確率：被正確識別為成員的比例
    attack_accuracy = float(forget_predictions.mean())
    
    # 平均置信度：模型認為遺忘集是成員的平均置信度
    average_confidence = float(forget_probs.mean())
    
    # 遺忘效果：遺忘集被識別為成員的比例(應該接近0.5)
    forget_effect = float(1 - abs(attack_accuracy - 0.5) * 2)
    
    return {
        'attack_accuracy': attack_accuracy,
        'average_confidence': average_confidence,
        'forget_effect': forget_effect,
        'total_samples': int(len(forget_probs))
    }


def comprehensive_unlearning_evaluation(original_model: nn.Module,
                                        unlearned_model: nn.Module,
                                        gold_model: nn.Module,
                                        random_model: nn.Module,
                                        retain_loader: DataLoader,
                                        forget_loader: DataLoader,
                                        test_loader: DataLoader,
                                        device: str = "cuda"):
    """
    全面評估遺忘效果
    
    Args:
        original_model: 原始模型
        unlearned_model: 遺忘後的模型
        gold_model: 黃金標準模型
        retain_loader: 保留數據加載器
        forget_loader: 遺忘數據加載器
        test_loader: 測試數據加載器
        device: 計算設備
    
    Returns:
        評估結果字典
    """
    results = {}
    
    # 計算ZRF分數（與隨機初始化模型比較）
    zrf = calc_zrf(unlearned_model, random_model, forget_loader, device)
    results['zrf_score'] = zrf
    
    # AlignmentScore（與 Gold/重訓比較）
    results['alignment_score'] = alignment_score(unlearned_model, gold_model, forget_loader, device)

    # 原始模型的MIA
    results['original_mia'] = get_membership_attack_scores(
        retain_loader, forget_loader, test_loader, original_model
    )
    
    # 遺忘模型的MIA
    results['unlearned_mia'] = get_membership_attack_scores(
        retain_loader, forget_loader, test_loader, unlearned_model
    )
    
    # 黃金標準模型的MIA
    results['gold_mia'] = get_membership_attack_scores(
        retain_loader, forget_loader, test_loader, gold_model
    )
    
    # MIA改進程度
    results['mia_improvement'] = (
        results['original_mia']['attack_accuracy'] - 
        results['unlearned_mia']['attack_accuracy']
    )
    
    # 遺忘成功度（綜合指標）
    results['unlearning_success'] = {
        'zrf_component': results['zrf_score'],
        'alignment_component': results['alignment_score'],
        # 'mia_component': results['unlearned_mia']['forget_effect'],
        'overall_score': (results['zrf_score'] + results['alignment_score']) / 2.0
        # 'overall_score': (results['zrf_score'] + results['alignment_score'] + results['unlearned_mia']['forget_effect']) / 3.0
    }
    return results


def print_evaluation_report(results):
    """
    打印評估報告
    
    Args:
        results: comprehensive_unlearning_evaluation返回的結果
    """
    print("\n" + "="*50)
    print("機器遺忘評估報告")
    print("="*50)
    
    print(f"\nZRF（vs Random）: {results['zrf_score']:.4f}  (越接近1越像「完全沒學過」)")
    print(f"AlignmentScore（vs Gold）: {results['alignment_score']:.4f}  (越接近1越接近重訓基線)")

    print("\n成員推斷攻擊結果:")
    print(f"原始模型 - 攻擊準確率: {results['original_mia']['attack_accuracy']:.4f}")
    print(f"遺忘模型 - 攻擊準確率: {results['unlearned_mia']['attack_accuracy']:.4f}")
    print(f"黃金標準 - 攻擊準確率: {results['gold_mia']['attack_accuracy']:.4f}")
    
    print(f"\nMIA改進程度: {results['mia_improvement']:.4f}")
    
    print("\n遺忘成功度評估:")
    print(f"ZRF成分: {results['unlearning_success']['zrf_component']:.4f}")
    print(f"Alignment: {results['unlearning_success']['alignment_component']:.4f}")
    print(f"MIA成分: {results['unlearning_success']['mia_component']:.4f}")
    print(f"綜合分數: {results['unlearning_success']['overall_score']:.4f}")
    
    print("\n建議:")
    if results['unlearning_success']['overall_score'] > 0.8:
        print("遺忘效果非常好!")
    elif results['unlearning_success']['overall_score'] > 0.6:
        print("遺忘效果良好，但仍有改進空間")
    else:
        print("遺忘效果不理想，建議調整遺忘策略")
    
    print("="*50)