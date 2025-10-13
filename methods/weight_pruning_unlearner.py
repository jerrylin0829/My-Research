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
                fine_tune_epochs=15,
                prune_mode='reset'
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
        
        if prune_strategy == 'magnitude_zero_lock':
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
            self._fine_tune_after_pruning(retain_loader, fine_tune_epochs)
        
        return self.model
    
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
            # if not isinstance(module, torch.nn.Linear):
            #     continue
            # if self._should_skip_param(name, module):
            #     continue
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
            if batch_count >= 10:  # 只用前10個batch計算，節省時間
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
    
    def _should_prune_layer(self, layer_name, target_layers):
        """判斷是否應該修剪該層"""
        for target in target_layers:
            if target == 'head' and 'head' in layer_name:
                return True
            # === 新增 'late_blocks_mlp' 目標：只針對後半部的 Block (5-9) MLP 層 ===
            elif target == 'late_blocks_mlp' and 'blocks.' in layer_name:
                try:
                    block_idx = int(layer_name.split('blocks.')[1].split('.')[0])
                    is_late_block = block_idx >= 5
                    is_mlp = ('mlp.fc1' in layer_name or 'mlp.fc2' in layer_name)
                    if is_late_block and is_mlp:
                        return True
                except:
                    return False
            # ====================================================================
            
            # === 現有 'blocks_mlp' 目標：針對所有 Block 的 MLP 層 ===
            elif target == 'blocks_mlp' and 'blocks.' in layer_name and ('mlp.fc1' in layer_name or 'mlp.fc2' in layer_name):
                return True
            # =======================================================
            elif target == 'late_blocks' and 'blocks' in layer_name:
                # 只修剪後半部分的blocks (包括 Attention 和 MLP)
                try:
                    if 'blocks.' in layer_name:
                        block_idx = int(layer_name.split('blocks.')[1].split('.')[0])
                        return block_idx >= 5  # 修剪第5層（索引 5）之後的blocks (5, 6, 7, 8, 9)
                except:
                    return False
            elif target == '30%_blocks' and 'blocks' in layer_name:
                # 只修剪後30%的blocks（假設總共10層）
                try:
                    if 'blocks.' in layer_name:
                        block_idx = int(layer_name.split('blocks.')[1].split('.')[0])
                        return block_idx >= 7  # 修剪第7層（索引 7）之後的blocks (7, 8, 9)
                except:
                    return False
            elif target == 'all_blocks' and 'blocks' in layer_name:
                return True
            elif target in layer_name:
                return True
        return False
    
    def _should_skip_param(self, name, module):
        # 跳過 LayerNorm
        if isinstance(module, torch.nn.LayerNorm):
            return True
        # 跳過位置/cls token/pos_embed
        if 'pos_embed' in name or 'cls_token' in name:
            return True
        # 跳過 patch_embed（token 化底座）
        if 'patch_embed' in name:
            return True
        # 跳過 bias
        #   - module 可能沒有 bias 或 bias 是 None
        if hasattr(module, 'bias') and module.bias is not None:
            # 若當前目標是 weight，只要這層有 bias，我們仍然只 prune weight；
            # 但為避免誤動 bias，可明確在下面的層內檢查 name.endswith('.bias')
            pass
        return False

    # def _fine_tune_after_pruning(self, retain_loader, epochs):
    #     """修剪後的微調"""
    #     print(f"修剪後微調 {epochs} 輪...")
        
    #     # 只訓練未被完全修剪的參數
    #     trainable_params = []
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             trainable_params.append(param)

    #     if not trainable_params:
    #         print("警告：沒有可訓練參數，跳過微調")
    #         return
        
    #     optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
    #     for epoch in range(epochs):
    #         self.model.train()
    #         total_loss = 0
            
    #         pbar = tqdm(retain_loader, desc=f"微調 Epoch {epoch+1}/{epochs}")
    #         for inputs, labels in pbar:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
                
    #             optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             loss = self.criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
                
    #             total_loss += loss.item()
    #             pbar.set_postfix({'loss': loss.item()})
            
    #         scheduler.step()
            
    #         # 評估
    #         val_acc = self._evaluate_retain(self.model, retain_loader)
    #         print(f"微調 Epoch {epoch+1}: Loss={total_loss/len(retain_loader):.4f}, Val_Acc={val_acc:.2f}%")

    def _fine_tune_after_pruning(self, retain_loader, epochs):
        """
        修剪後的微調 - 導入 OneCycleLR 和改進的優化器設定
        """
        print(f"修剪後微調 {epochs} 輪...")
        
        # 設置超參數 (可以透過 argparse 傳入，這裡使用固定值作為示範)
        FT_LR = 3e-4          # 略高於預設的 1e-4，確保收斂速度
        FT_WEIGHT_DECAY = 0.05
        
        # 只訓練未被完全修剪的參數（即 `param.requires_grad` 為 True 的所有參數）
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        if not trainable_params:
            print("警告：沒有可訓練參數，跳過微調")
            return
        
        # 1. 優化器：使用 AdamW，略高的 Weight Decay 有助於正則化
        optimizer = optim.AdamW(trainable_params, lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
        
        # 2. 學習率調度器：使用 OneCycleLR (最快、最有效率的微調調度器)
        # 為了使用 OneCycleLR，我們需要知道總訓練步數
        steps_per_epoch = len(retain_loader)
        
        # 由於 FT_LR 只是初始 LR，OneCycleLR 的 max_lr 設為 10 倍
        max_lr = FT_LR * 10
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            anneal_strategy='cos',
            pct_start=0.3
        )
        
        # 3. 損失函數：引入 Label Smoothing 正則化，並可考慮加入 Logit 懲罰
        # Label Smoothing: 避免過度自信，有助於提高泛化能力
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 記錄最佳準確度
        best_acc = self._evaluate_retain(self.model, retain_loader)
        best_state = self.model.state_dict().copy() # 避免在 CUDA 之間移動，直接 copy state_dict
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(retain_loader, desc=f"微調 Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # === 損失計算 ===
                loss = criterion(outputs, labels)
                
                # 可選正則化：Logit L2 Norm Penalty
                # if epoch < epochs // 2: # 僅在前一半輪次施加懲罰
                #     logit_norm = outputs.norm(p=2, dim=1).mean()
                #     lambda_logit = 1e-5
                #     loss += lambda_logit * logit_norm

                loss.backward()
                optimizer.step()
                
                # ⚠️ OneCycleLR 必須每批次 step 一次
                scheduler.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            
            # 評估 (使用 mul_vit_1.py 中的 _evaluate_retain 來確保一致性)
            # 💡 這裡呼叫的是 Unlearner 內的 _evaluate_retain，需要傳入 loader
            val_acc = self._evaluate_retain(self.model, retain_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"微調 Epoch {epoch+1}: Loss={total_loss/len(retain_loader):.4f}, Val_Acc={val_acc:.2f}%, LR={current_lr:.6f}")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                # 只保存 state_dict 的 CPU 版本以節省顯存
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # 載入最佳模型狀態（這對於 reset 模式的模型穩定性非常重要）
        self.model.load_state_dict(best_state)
        print(f"微調完成，恢復最佳模型 (Retain Acc: {best_acc:.2f}%)")
    
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

            configs[f'magnitude_zero_lock_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'magnitude_zero_lock',
                'prune_mode': 'zero_lock',
                'target_layers': target_layers,
            }

            configs[f'magnitude_reset_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'magnitude_reset',
                'prune_mode': 'reset',       
                'target_layers': target_layers,
            }
        
        # Gradient-based strategies  
        for ratio in prune_ratios:
            configs[f'gradient_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'gradient',
                'target_layers': target_layers,
            }
        
        # Fisher-based strategies
        for ratio in prune_ratios:
            configs[f'fisher_{int(ratio*100)}%'] = {
                'prune_ratio': ratio,
                'prune_strategy': 'fisher',
                'target_layers': target_layers,
            }
        
        return configs
