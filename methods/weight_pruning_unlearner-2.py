"""
WeightPruning 遺忘器 - 智能權重修剪策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_unlearner import BaseUnlearner


class WeightPruningUnlearner(BaseUnlearner):
    """
    權重修剪遺忘：移除與遺忘數據相關的權重
    
    用途與意義：
    1. 直接移除與遺忘數據相關的神經連接
    2. 研究權重重要性與遺忘的關係
    3. 實現永久性的知識移除
    """
    
    def unlearn(self, 
                retain_loader, 
                forget_loader, 
                prune_ratio=0.1, 
                prune_strategy='magnitude_reset', 
                target_layers=['head', 'late_blocks'],
                fine_tune_epochs=20,
                prune_mode='reset',
                **ft_kwargs
    ):
        """
        權重修剪遺忘主方法
        
        Args:
            retain_loader: 保留數據加載器
            forget_loader: 遺忘數據加載器
            prune_ratio: 修剪比例
            prune_strategy: 修剪策略 ('magnitude', 'gradient', 'fisher')
            target_layers: 目標層
            fine_tune_epochs: 修剪後微調輪數
        """
        
        print(f"開始權重修剪遺忘，策略: {prune_strategy}, 修剪比例: {prune_ratio}")
        print("保持原始 100 類分類頭，不進行類別映射")
        
        # 給 L2-SP（目前不啟用）預留的原始拷貝
        self._orig_sd = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

        if prune_strategy == 'magnitude_zero_lock':
            # self._magnitude_based_pruning(prune_ratio, target_layers, add_noise = False)
            # if prune_mode == 'zero_lock':
            self._magnitude_prune_zero_lock(prune_ratio, target_layers)
        elif prune_strategy == 'magnitude_reset':
            self._magnitude_prune_reset(prune_ratio, target_layers)
        elif prune_strategy == 'gradient':
            self._gradient_based_pruning(retain_loader, forget_loader, prune_ratio, target_layers)
        elif prune_strategy == 'fisher':
            self._fisher_based_pruning(retain_loader, prune_ratio, target_layers)
        else:
            raise ValueError(f"不支援的修剪策略: {prune_strategy}")
        
        # 修剪後微調
        if retain_loader is not None and fine_tune_epochs > 0:
            self._fine_tune_after_pruning(retain_loader, fine_tune_epochs, **ft_kwargs)
        
        return self.model
    
    def _magnitude_based_pruning(self, 
                                 prune_ratio, 
                                 target_layers, 
                                 add_noise=False, 
                                 noise_std=1e-2):
        """
        基於權重大小的修剪
        新增雜訊而非直接置零
        """
        print("執行基於權重大小的修剪...")
        pruned_count = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if self._should_prune_layer(name, target_layers) and hasattr(module, 'weight'):
                weight = module.weight.data
                total_params += weight.numel()
                
                # 計算權重的重要性分數（絕對值）
                importance = torch.abs(weight)
                
                # 計算閾值（保留最重要的權重）
                threshold = torch.quantile(importance.flatten(), prune_ratio)
                
                # 創建掩碼（小於閾值的權重被修剪）
                mask = importance >= threshold
                pruning_mask = importance < threshold

                if add_noise:
                    # 🛠️ 添加小量隨機噪聲而非置零
                    noise_scale = weight[mask].std() * noise_std
                    noise = torch.randn_like(weight[pruning_mask]) * noise_scale
                    module.weight.data[pruning_mask] = noise
                else:
                    module.weight.data *= mask.float()
                
                with torch.no_grad():
                    if weight.dim() >= 2:
                        # 2D或更高維度：使用fan_in
                        fan_in = weight.size(-1)  # 最後一維作為fan_in
                        std = (2.0 / fan_in) ** 0.5
                    else:
                        # 1D權重：使用當前權重的標準差
                        std = weight.std().item() if weight.numel() > 1 else 0.01
                    
                    # 重新初始化被修剪的權重
                    module.weight.data[~mask] = torch.randn_like(
                        module.weight.data[~mask]
                    ) * std * 0.1

                # 統計修剪數量
                pruned_count += pruning_mask.sum().item()
                
                print(f"修剪層 {name}: {pruning_mask.sum().item()}/{weight.numel()} "
                      f"({100 * pruning_mask.sum().item() / weight.numel():.1f}%)")

        print(f"總共修剪參數: {pruned_count:,}/{total_params:,} ({100*pruned_count/total_params:.1f}%)")
    
    def _magnitude_prune_reset(self, prune_ratio, target_layers):
        """
        reset 版：小權重 => 直接置 0，不掛 mask，後續可被梯度更新
        """
        print("[Pruning] reset mode (no mask; weights can grow again)")
        pruned_total = 0
        total_params = 0

        for name, module in self.model.named_modules():
            if not self._should_prune_layer(name, target_layers): 
                continue
            if not hasattr(module, 'weight'):
                continue

            W = module.weight.data
            total_params += W.numel()

            imp = torch.abs(W)
            thr = torch.quantile(imp.flatten(), prune_ratio)
            keep_mask = (imp >= thr)
            prune_mask = ~keep_mask

            # 1) 直接把被剪位置設為 0（不掛任何 mask）
            W[prune_mask] = 0.0

            pruned_total += prune_mask.sum().item()
            ratio = 100 * prune_mask.float().mean().item()
            print(f"修剪層 {name}: {int(prune_mask.sum())}/{W.numel()} ({ratio:.1f}%)")

        print(f"總共修剪參數: {pruned_total:,}/{total_params:,} ({100*pruned_total/max(total_params,1):.1f}%)")
        
    def _magnitude_prune_zero_lock(self, prune_ratio, target_layers, noise_std=0.0):
        """
        magnitude 決定遮罩 + torch.prune 掛 reparam（不 remove）
        ➜ forward/ backward 永遠用 weight_orig * mask，被剪位置真 0 且無梯度，"不會長回來"
        """
        print("[Pruning] zero_lock with reparam (no remove)")
        pruned_total = 0
        total_params = 0

        for name, module in self.model.named_modules():
            if not self._should_prune_layer(name, target_layers):
                continue
            if not hasattr(module, 'weight'):
                continue

            W = module.weight.data
            total_params += W.numel()

            # 1) importance & threshold
            imp = torch.abs(W)
            thr = torch.quantile(imp.flatten(), prune_ratio)
            keep_mask = (imp >= thr).to(W.dtype)  # 1=keep, 0=prune
            prune_mask = (imp < thr)

            # 2) 先把被剪位置設 0（方便觀察）
            W.mul_(keep_mask)

            # 3) 掛上 reparam 遮罩（不 remove）
            prune.custom_from_mask(module, name='weight', mask=keep_mask)

            # 4)（可選）只對保留位置加很小噪聲穩定訓練
            if noise_std and noise_std > 0:
                with torch.no_grad():
                    std = W[keep_mask.bool()].std().clamp_min(1e-8)
                    noise = torch.randn_like(W) * std * noise_std
                    W.add_(noise * keep_mask)

            pruned_total += prune_mask.sum().item()
            print(f"修剪層 {name}: {int(prune_mask.sum())}/{W.numel()} ({100*prune_mask.float().mean().item():.1f}%)")

        print(f"總共修剪參數: {pruned_total:,}/{total_params:,} ({100*pruned_total/max(total_params,1):.1f}%)")
        
    def _gradient_based_pruning(self, retain_loader, forget_loader, prune_ratio, target_layers):
        """基於梯度的修剪：移除對遺忘數據梯度大但對保留數據梯度小的權重"""
        print("執行基於梯度的修剪...")
        
        # 計算對遺忘數據的梯度
        forget_gradients = self._compute_gradients(forget_loader, "遺忘數據")
        
        # 計算對保留數據的梯度
        retain_gradients = self._compute_gradients(retain_loader, "保留數據")
        
        pruned_count = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if (self._should_prune_layer(name, target_layers) and 
                hasattr(module, 'weight') and 
                name in forget_gradients and 
                name in retain_gradients):
                
                forget_grad = forget_gradients[name]
                retain_grad = retain_gradients[name]
                
                # 計算遺忘分數：對遺忘數據敏感但對保留數據不敏感的權重
                forget_score = torch.abs(forget_grad) / (torch.abs(retain_grad) + 1e-8)
                
                # 修剪分數最高的權重
                threshold = torch.quantile(forget_score.flatten(), 1 - prune_ratio)
                mask = forget_score < threshold
                
                # 統計
                total_params += module.weight.numel()
                pruned_count += (mask == 0).sum().item()
                
                # 應用掩碼
                module.weight.data *= mask.float()
                
                print(f"梯度修剪層 {name}: {(mask == 0).sum().item()}/{module.weight.numel()} "
                      f"({100 * (mask == 0).sum().item() / module.weight.numel():.1f}%)")
        
        print(f"總共修剪參數: {pruned_count:,}/{total_params:,} ({100*pruned_count/total_params:.1f}%)")
    
    def _fisher_based_pruning(self, retain_loader, prune_ratio, target_layers):
        """基於Fisher信息的修剪：保留對保留數據重要的權重"""
        print("執行基於Fisher信息的修剪...")
        
        fisher_info = self._compute_fisher_information(retain_loader)
        
        pruned_count = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if (self._should_prune_layer(name, target_layers) and 
                hasattr(module, 'weight') and 
                name in fisher_info):
                
                fisher = fisher_info[name]
                
                # 保留Fisher信息最大的權重
                threshold = torch.quantile(fisher.flatten(), prune_ratio)
                mask = fisher > threshold
                
                # 統計
                total_params += module.weight.numel()
                pruned_count += (mask == 0).sum().item()
                
                # 應用掩碼
                module.weight.data *= mask.float()
                
                print(f"Fisher修剪層 {name}: {(mask == 0).sum().item()}/{module.weight.numel()} "
                      f"({100 * (mask == 0).sum().item() / module.weight.numel():.1f}%)")
        
        print(f"總共修剪參數: {pruned_count:,}/{total_params:,} ({100*pruned_count/total_params:.1f}%)")
    
    def _compute_gradients(self, dataloader, data_type):
        """計算對特定數據集的梯度"""
        print(f"計算{data_type}的梯度...")
        gradients = {}
        self.model.eval()
        
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 2:  # 只用前2個batch計算，節省時間
                break
                
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    if name not in gradients:
                        gradients[name] = torch.zeros_like(module.weight)
                    gradients[name] += torch.abs(module.weight.grad)
            
            batch_count += 1
        
        # 平均化梯度
        for name in gradients:
            gradients[name] /= batch_count
        
        return gradients
    
    def _compute_fisher_information(self, dataloader):
        """計算Fisher信息矩陣"""
        print("計算Fisher信息...")
        fisher_info = {}
        self.model.eval()
        
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 3:  # 只用前3個batch計算
                break
                
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            
            # 對每個類別計算Fisher信息
            for c in range(outputs.size(1)):
                self.model.zero_grad()
                class_loss = outputs[:, c].sum()
                class_loss.backward(retain_graph=True)
                
                for name, module in self.model.named_modules():
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        if name not in fisher_info:
                            fisher_info[name] = torch.zeros_like(module.weight)
                        fisher_info[name] += module.weight.grad ** 2
            
            batch_count += 1
        
        # 平均化Fisher信息
        for name in fisher_info:
            fisher_info[name] /= (batch_count * outputs.size(1))
        
        return fisher_info
    
    def _build_scheduler(self, optimizer, scheduler_type, epochs, train_loader, min_lr):
        if scheduler_type == "cosine_warmup":
            total_steps = max(1, epochs * len(train_loader))
            warmup_steps = max(1, int(0.1 * total_steps))  # 前 10% steps 做 warmup
            base_lr = optimizer.param_groups[0]['lr']
            floor = float(min_lr) / float(base_lr)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return floor + (1.0 - floor) * cosine

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        else:
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    def _should_prune_layer(self, layer_name, target_layers):
        """判斷是否應該修剪該層"""
        for target in target_layers:
            if target == 'head' and 'head' in layer_name:
                return True
            elif target == 'late_blocks' and 'blocks' in layer_name:
                # 只修剪後半部分的blocks
                try:
                    if 'blocks.' in layer_name:
                        block_idx = int(layer_name.split('blocks.')[1].split('.')[0])
                        return block_idx >= 5  # 修剪第5層之後的blocks
                except:
                    return False
            elif target == '30%_blocks' and 'blocks' in layer_name:
                # 只修剪後30%的blocks（假設總共10層）
                try:
                    if 'blocks.' in layer_name:
                        block_idx = int(layer_name.split('blocks.')[1].split('.')[0])
                        return block_idx >= 7  # 修剪第7層之後的blocks
                except:
                    return False
            elif target == 'all_blocks' and 'blocks' in layer_name:
                return True
            elif target in layer_name:
                return True
            
            # 命中目標層之後，再檢查是否該跳過（LayerNorm / bias / 1D 向量）
            # 這裡用名字快速略過；實際套用在 named_modules() 時會只對有 .weight 的 module 生效
            # if ('norm' in layer_name) or ('ln' in layer_name) or ('bias' in layer_name):
            #     return False
        
        return False
    
    
    def _fine_tune_after_pruning(
            self,
            retain_loader,
            epochs=15,
            lr=3e-4,
            weight_decay=0.05,
            scheduler_type="cosine_warmup",
            min_lr=1e-6,
            use_ema=True,
            ema_decay=0.999,
            use_l2sp=False,
            l2sp_lambda=5e-4,
            l2sp_layers=("head", "blocks.7", "blocks.8", "blocks.9")  # 依你 ViT depth=10
        ):
        """
        修剪(reset/zero-lock)後的穩定微調：
        - AdamW + CosineWarmup / Cosine
        - 可選 EMA（評估/挑 best 用 EMA 權重）
        - 可選 L2-SP：僅約束 head + late blocks 朝原模型權重靠攏
        - 早停依 retain_acc
        """
        print(f"修剪後微調 {epochs} 輪... (lr={lr}, wd={weight_decay}, sched={scheduler_type}, ema={use_ema}, l2sp={use_l2sp})")

        # 1) 準備可訓練參數
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            print("警告：沒有可訓練參數，跳過微調")
            return

        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 2) 調度器（與 GS 一致邏輯）
        scheduler = self._build_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            epochs=epochs,
            train_loader=retain_loader,
            min_lr=min_lr
        )

        # 3) EMA（僅在參數需訓練時啟用）
        ema_shadow = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad} if use_ema else None
        def ema_update():
            if ema_shadow is None: return
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    ema_shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

        def ema_apply():
            if ema_shadow is None: return None
            snap = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            sd = self.model.state_dict()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    sd[n].copy_(ema_shadow[n])
            self.model.load_state_dict(sd)
            return snap

        def ema_restore(snapshot):
            if snapshot is None: return
            self.model.load_state_dict(snapshot)

        # 4) 供 L2-SP 使用的原始權重（請在 unlearn() 一開始先緩存：self._orig_sd = clone_state_dict(self.model)）
        orig_sd = getattr(self, "_orig_sd", None)

        def l2sp_penalty():
            if (not use_l2sp) or (orig_sd is None):
                return 0.0
            pen = 0.0
            for n, p in self.model.named_parameters():
                if (not p.requires_grad):
                    continue
                if any(tag in n for tag in l2sp_layers):
                    pen = pen + (p - orig_sd[n].to(p.device)).pow(2).sum()
            return pen * l2sp_lambda

        # 5) 早停
        best_acc = -1.0
        best_state = None
        patience = 6
        no_improve = 0

        for ep in range(epochs):
            self.model.train()
            total_loss = 0.0
            total = 0
            correct = 0

            pbar = tqdm(retain_loader, desc=f"微調 Epoch {ep+1}/{epochs}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x)
                ce_loss = criterion(logits, y)

                # 熱度/校準：小熵正則（抑制過度自信）
                probs = F.softmax(logits, dim=1)
                entropy_reg = -(probs * (probs + 1e-12).log()).sum(dim=1).mean()
                loss = ce_loss + 1e-3 * entropy_reg

                # 幾何穩定：L2-SP（僅 head+late blocks）
                l2sp = l2sp_penalty()
                if l2sp != 0.0:
                    loss = loss + l2sp

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler_type in {"onecycle", "cosine_warmup"}:
                    scheduler.step()

                # EMA
                ema_update()

                with torch.no_grad():
                    pred = logits.argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                    total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{100*correct/max(total,1):.2f}%"})

            train_loss = total_loss / max(1, len(retain_loader))
            train_acc = 100.0 * correct / max(1, total)

            # 用 EMA 權重驗證 → 再還原
            snapshot = ema_apply()
            val_acc = self._evaluate_retain(self.model, retain_loader)
            ema_restore(snapshot)

            if scheduler_type in {"cosine", "step"}:
                scheduler.step()
            # plateau 的話加上 elif scheduler_type == "plateau": scheduler.step(val_acc)

            print(f"微調 Epoch {ep+1}: Loss={train_loss:.4f}, Train_Acc={train_acc:.2f}%, Val_Acc={val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
                no_improve = 0
                print(f"  [New Best] retain_acc={val_acc:.2f}%")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"早停：{patience} 個 epoch 無提升")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"微調完成，best retain_acc={best_acc:.2f}%")

    def _evaluate_retain(self, model, retain_loader):
        """評估保留集準確率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def get_strategy_configs(self, prune_ratios=None, target_layers=None):
        """獲取所有可用的策略配置"""
        configs = {}
        
        if prune_ratios is None or len(prune_ratios) == 0:
            prune_ratios = [0.05, 0.1, 0.2, 0.3]
        
        # Magnitude-based strategies
        for ratio in prune_ratios:
            # configs[f'magnitude_{int(ratio*100)}%'] = {
            #     'prune_ratio': ratio,
            #     'prune_strategy': 'magnitude',
            #     # 'target_layers': ['head', 'all_blocks'],
            #     'target_layers': target_layers,
            # }

            configs[f'magnitude_zero_lock_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'magnitude_zero_lock',
                'prune_mode': 'zero_lock',
                'target_layers': target_layers,
                # 'target_layers': ['head', 'all_blocks']
            }

            configs[f'magnitude_reset_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'magnitude_reset',
                'prune_mode': 'reset',        # ← reset 版
                'target_layers': target_layers,
                # 'target_layers': ['head', 'all_blocks']
            }
        
        # Gradient-based strategies  
        for ratio in prune_ratios:
            configs[f'gradient_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'gradient',
                # 'target_layers': ['head', 'all_blocks'],
                'target_layers': target_layers,
            }
        
        # Fisher-based strategies
        for ratio in prune_ratios:
            configs[f'fisher_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'fisher',
                # 'target_layers': ['head', 'all_blocks'],
                'target_layers': target_layers,
            }
        
        return configs
