"""
增強評估工具 - 包含所有新的評估指標
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from metrics import calc_kldscore,  calc_zrf, alignment_score
from scipy import stats


def cosine_similarity_score(model_a, model_b, dataloader, device='cuda'):
    """計算兩個模型輸出的餘弦相似度"""
    model_a.eval()
    model_b.eval()
    
    similarities = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            
            out_a = F.normalize(model_a(x), p=2, dim=1)
            out_b = F.normalize(model_b(x), p=2, dim=1)
            
            similarity = F.cosine_similarity(out_a, out_b, dim=1)
            similarities.append(similarity.cpu())
    
    return torch.cat(similarities).mean().item()


def activation_distance(model_a, model_b, dataloader, device='cuda', layer_name='blocks.-1'):
    """計算特定層的激活距離"""
    activations_a = []
    activations_b = []
    
    def hook_fn_a(module, input, output):
        activations_a.append(output.detach().cpu())
    
    def hook_fn_b(module, input, output):
        activations_b.append(output.detach().cpu())
    
    # 註冊hooks - 簡化版本，選擇最後一個transformer block
    target_a = None
    for name, module in model_a.named_modules():
        if name.endswith('mlp') and name.startswith('blocks.'):
            target_a = module
    target_b = None
    for name, module in model_b.named_modules():
        if name.endswith('mlp') and name.startswith('blocks.'):
            target_b = module

    if target_a is None or target_b is None:
        print("警告：未找到最後一層 MLP")
        return 0.0
    
    hook_a = target_a.register_forward_hook(hook_fn_a)
    hook_b = target_b.register_forward_hook(hook_fn_b)

    model_a.eval(); model_b.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            _ = model_a(x); _ = model_b(x)

    hook_a.remove(); hook_b.remove()

    if activations_a and activations_b:
        a = torch.cat(activations_a, dim=0)
        b = torch.cat(activations_b, dim=0)
        n = min(a.size(0), b.size(0))
        return F.mse_loss(a[:n], b[:n]).item()
    return 0.0


def statistical_distance_score(model_a, model_b, dataloader, device='cuda'):
    """計算輸出分佈的統計距離"""
    predictions_a = []
    predictions_b = []
    
    model_a.eval()
    model_b.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            
            pred_a = F.softmax(model_a(x), dim=1)
            pred_b = F.softmax(model_b(x), dim=1)
            
            predictions_a.append(pred_a.cpu().numpy())
            predictions_b.append(pred_b.cpu().numpy())
    
    if not predictions_a or not predictions_b:
        return {'total_variation': 0.0, 'js_divergence': 0.0}
    
    pred_a = np.concatenate(predictions_a, axis=0)
    pred_b = np.concatenate(predictions_b, axis=0)
    
    # 計算多種統計距離
    results = {}
    
    # 總變分距離
    tv_distance = 0.5 * np.sum(np.abs(pred_a - pred_b), axis=1).mean()
    results['total_variation'] = tv_distance
    
    # JS散度（簡化計算）
    epsilon = 1e-8
    pred_a_safe = pred_a + epsilon
    pred_b_safe = pred_b + epsilon
    
    m = 0.5 * (pred_a_safe + pred_b_safe)
    js_div = 0.5 * np.sum(pred_a_safe * np.log(pred_a_safe / m), axis=1).mean() + \
             0.5 * np.sum(pred_b_safe * np.log(pred_b_safe / m), axis=1).mean()
    results['js_divergence'] = js_div
    
    return results


def feature_attribution_analysis(model, forget_loader, retain_loader, device='cuda'):
    """簡化的特徵歸因分析"""
    model.eval()
    
    forget_gradients = []
    retain_gradients = []
    
    # 計算遺忘集的梯度範數
    for batch_idx, batch in enumerate(forget_loader):
        if batch_idx >= 3:  # 只用前3個batch
            break
        
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        # 計算對輸入的梯度
        for i in range(min(5, outputs.size(0))):  # 每個batch只取5個樣本
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[i, outputs[i].argmax()] = 1.0
            
            grads = torch.autograd.grad(
                outputs, inputs, grad_outputs=grad_outputs,
                create_graph=False, retain_graph=True
            )[0]
            
            forget_gradients.append(grads[i].abs().mean().item())
    
    # 計算保留集的梯度範數
    for batch_idx, batch in enumerate(retain_loader):
        if batch_idx >= 3:  # 只用前3個batch
            break
        
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        for i in range(min(5, outputs.size(0))):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[i, outputs[i].argmax()] = 1.0
            
            grads = torch.autograd.grad(
                outputs, inputs, grad_outputs=grad_outputs,
                create_graph=False, retain_graph=True
            )[0]
            
            retain_gradients.append(grads[i].abs().mean().item())
    
    # 計算統計
    forget_mean = np.mean(forget_gradients) if forget_gradients else 0
    retain_mean = np.mean(retain_gradients) if retain_gradients else 0
    
    return {
        'forget_attribution': forget_mean,
        'retain_attribution': retain_mean,
        'attribution_ratio': forget_mean / (retain_mean + 1e-8)
    }


def enhanced_unlearning_evaluation(original_model, unlearned_model, gold_model, random_model,
                                   retain_loader, forget_loader, test_loader, device):
    """增強版綜合評估"""
    print("執行增強版綜合評估...")
    results = {}

    # ZRF（vs Random Init）
    try:
        print("計算 ZRF 分數（vs Random）...")
        results['zrf_score'] = calc_zrf(unlearned_model, random_model, forget_loader, device)
        print(f"ZRF 分數: {results['zrf_score']:.4f}")
    except Exception as e:
        print(f"❌ ZRF 計算錯誤: {e}")
        results['zrf_score'] = 0.0

    # Alignment（vs Gold）
    try:
        print("計算 Alignment 分數（vs Gold）...")
        results['alignment_score'] = alignment_score(unlearned_model, gold_model, forget_loader, device)
        print(f"Alignment 分數: {results['alignment_score']:.4f}")
    except Exception as e:
        print(f"❌ Alignment 計算錯誤: {e}")
        results['alignment_score'] = 0.0

    return results


def calculate_model_size_difference(original_model, unlearned_model):
    """計算模型大小變化（考慮 torch.nn.utils.prune 的 weight_orig / weight_mask）"""

    def count_nonzero_effective_weight(module):
        """
        回傳 (total_params, nonzero_params) for a module's weight
        - 若存在 weight_orig + weight_mask：以 (weight_orig * weight_mask) 計非零
        - 否則若存在 weight：以 weight 計非零
        - 否則回傳 (0, 0)
        """
        total = nonzero = 0
        # 直接到 submodule 層級找，不遞迴
        pnames = dict(module.named_parameters(recurse=False))
        bnames = dict(module.named_buffers(recurse=False))

        if 'weight_orig' in pnames and 'weight_mask' in bnames:
            w = pnames['weight_orig'].detach()
            m = bnames['weight_mask'].detach()
            eff = w * m
            total = eff.numel()
            nonzero = torch.count_nonzero(eff).item()
        elif 'weight' in pnames:
            w = pnames['weight'].detach()
            total = w.numel()
            nonzero = torch.count_nonzero(w).item()
        return total, nonzero

    def summarize_model(model, title):
        print(title)
        total_all, nonzero_all = 0, 0
        for name, module in model.named_modules():
            # 只列出有 weight 的模組
            t, nz = count_nonzero_effective_weight(module)
            if t > 0:
                print(f"層 {name}.weight: {nz}/{t} 非零參數")
                total_all += t
                nonzero_all += nz
        print(f"總計: {nonzero_all}/{total_all}")
        return total_all, nonzero_all

    print("=== 模型參數統計 ===")
    orig_total, orig_nonzero = summarize_model(original_model, "原始模型:")
    unl_total,  unl_nonzero  = summarize_model(unlearned_model, "修剪後模型:")

    sparsity_increase = (orig_nonzero - unl_nonzero) / max(orig_nonzero, 1)
    compression_ratio = unl_nonzero / max(orig_nonzero, 1)

    print(f"\n稀疏度增加: {sparsity_increase:.4f} ({sparsity_increase*100:.2f}%)")
    print(f"壓縮比: {compression_ratio:.4f}")

    return {
        'original_params': orig_total,
        'original_nonzero': orig_nonzero,
        'unlearned_params': unl_total,
        'unlearned_nonzero': unl_nonzero,
        'sparsity_increase': sparsity_increase,
        'compression_ratio': compression_ratio
    }


def print_enhanced_evaluation_report(results):
    """打印增強評估報告"""
    print("\n" + "="*80)
    print("增強評估報告")
    print("="*80)
    
    print(f"ZRF（vs Random）: {results.get('zrf_score', 0):.4f}")
    print(f"Alignment（vs Gold）: {results.get('alignment_score', 0):.4f}") 
    print(f"KL散度: {results.get('kl_divergence', 0):.4f}")
    print(f"餘弦相似度: {results.get('cosine_similarity', 0):.4f}")
    print(f"激活距離: {results.get('activation_distance', 0):.4f}")
    
    stat_dist = results.get('statistical_distances', {})
    print(f"總變分距離: {stat_dist.get('total_variation', 0):.4f}")
    print(f"JS散度: {stat_dist.get('js_divergence', 0):.4f}")
    
    feat_attr = results.get('feature_attribution', {})
    print(f"特徵歸因比例: {feat_attr.get('attribution_ratio', 0):.4f}")
    
    if 'model_size' in results:
        size_info = results['model_size']
        print(f"模型稀疏度增加: {size_info.get('sparsity_increase', 0):.2%}")
        print(f"壓縮比: {size_info.get('compression_ratio', 1):.3f}")
    
    print("="*80)
