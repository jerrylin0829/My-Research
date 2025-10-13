"""
LayerWise 遺忘器 - 按層遺忘策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_unlearner import BaseUnlearner


class LayerWiseUnlearner(BaseUnlearner):
    """
    按層遺忘：針對不同層使用不同的遺忘策略
    
    用途與意義：
    1. 研究不同層對遺忘的敏感度
    2. 探索漸進式遺忘策略
    3. 平衡遺忘效果與計算效率
    """
    
    def unlearn(self, retain_loader, forget_loader=None, 
                unfreeze_layers=4, strategy='standard',
                epochs=30, lr=5e-5, class_mapping=None, num_retain_classes=90):
        """
        按層遺忘主方法
        
        Args:
            retain_loader: 保留數據加載器
            forget_loader: 遺忘數據加載器（可選）
            unfreeze_layers: 解凍的層數（從最後往前數）
            strategy: 遺忘策略 ('standard', 'progressive')
            epochs: 訓練輪數
            lr: 學習率
            class_mapping: 類別映射
            num_retain_classes: 保留類別數
        """
        
        print(f"開始LayerWise遺忘，策略: {strategy}, 解凍層數: {unfreeze_layers}")
        
        # 調整分類頭
        if class_mapping is not None:
            self._adjust_classification_head(num_retain_classes)
        
        if strategy == 'progressive':
            return self._progressive_unfreeze_strategy(
                retain_loader, epochs, lr, class_mapping
            )
        else:
            return self._standard_strategy(
                retain_loader, unfreeze_layers, epochs, lr, class_mapping
            )
    
    def _standard_strategy(self, retain_loader, unfreeze_layers, epochs, lr, class_mapping):
        """標準按層遺忘策略"""
        
        # 先凍結所有參數
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解凍分類頭
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        # 解凍指定數量的最後幾層
        if unfreeze_layers > 0:
            for block in self.model.blocks[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # 統計可訓練參數
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        
        print(f"可訓練參數: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.1f}%)")
        
        # 設置優化器
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 訓練循環
        best_acc = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(retain_loader, desc=f"LayerWise Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 處理類別映射
                if class_mapping is not None:
                    mapped_labels = []
                    mapped_inputs = []
                    for i, label in enumerate(labels):
                        if label.item() in class_mapping:
                            mapped_labels.append(class_mapping[label.item()])
                            mapped_inputs.append(inputs[i])
                    
                    if not mapped_labels:
                        continue
                    
                    labels = torch.tensor(mapped_labels, device=self.device)
                    inputs = torch.stack(mapped_inputs)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            scheduler.step()
            
            # 評估
            val_acc = self._evaluate_retain(self.model, retain_loader, class_mapping)
            print(f"Epoch {epoch+1}: Loss={total_loss/len(retain_loader):.4f}, Val_Acc={val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
        
        print(f"LayerWise遺忘完成，最佳準確率: {best_acc:.2f}%")
        return self.model
    
    def _progressive_unfreeze_strategy(self, retain_loader, epochs, lr, class_mapping):
        """漸進式解凍策略"""
        print("執行漸進式解凍策略...")
        
        # 定義解凍階段 (針對10個blocks)
        stages = [
            {'epochs_ratio': 0.25, 'unfreeze_blocks': 0, 'name': '只訓練head'},
            {'epochs_ratio': 0.25, 'unfreeze_blocks': 1, 'name': 'head + 最後1層'},
            {'epochs_ratio': 0.25, 'unfreeze_blocks': 2, 'name': 'head + 最後2層'},
            {'epochs_ratio': 0.25, 'unfreeze_blocks': 4, 'name': 'head + 最後4層'},
        ]
        
        best_acc = 0
        current_epoch = 0
        
        for stage_idx, stage in enumerate(stages):
            stage_epochs = max(1, int(epochs * stage['epochs_ratio']))
            print(f"\n階段 {stage_idx+1}: {stage['name']}，訓練 {stage_epochs} 輪")
            
            # 重新凍結所有參數
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 解凍head
            for param in self.model.head.parameters():
                param.requires_grad = True
            
            # 解凍指定數量的blocks
            if stage['unfreeze_blocks'] > 0:
                for block in self.model.blocks[-stage['unfreeze_blocks']:]:
                    for param in block.parameters():
                        param.requires_grad = True
            
            # 為當前階段設置優化器（使用較小的學習率）
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            stage_lr = lr * (0.8 ** stage_idx)  # 每階段學習率遞減
            optimizer = optim.AdamW(trainable_params, lr=stage_lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
            
            # 該階段的訓練
            for epoch in range(stage_epochs):
                self.model.train()
                total_loss = 0
                
                pbar = tqdm(retain_loader, desc=f"Stage {stage_idx+1} Epoch {epoch+1}/{stage_epochs}")
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # 處理類別映射
                    if class_mapping is not None:
                        mapped_labels = []
                        mapped_inputs = []
                        for i, label in enumerate(labels):
                            if label.item() in class_mapping:
                                mapped_labels.append(class_mapping[label.item()])
                                mapped_inputs.append(inputs[i])
                        
                        if not mapped_labels:
                            continue
                        
                        labels = torch.tensor(mapped_labels, device=self.device)
                        inputs = torch.stack(mapped_inputs)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
                
                scheduler.step()
                current_epoch += 1
            
            # 評估當前階段結果
            stage_acc = self._evaluate_retain(self.model, retain_loader, class_mapping)
            print(f"階段 {stage_idx+1} 完成，準確率: {stage_acc:.2f}%")
            
            if stage_acc > best_acc:
                best_acc = stage_acc
        
        print(f"漸進式遺忘完成，最佳準確率: {best_acc:.2f}%")
        return self.model
    
    def _adjust_classification_head(self, num_retain_classes):
        """調整分類頭以適應新的類別數量"""
        if hasattr(self.model, 'head') and self.model.head.out_features != num_retain_classes:
            old_head = self.model.head
            self.model.head = nn.Linear(old_head.in_features, num_retain_classes).to(self.device)
            print(f"分類頭已調整: {old_head.out_features} -> {num_retain_classes}")
    
    def get_strategy_configs(self):
        """獲取所有可用的策略配置"""
        return {
            'head_only': {'unfreeze_layers': 0, 'strategy': 'standard'},
            'head_plus_last1': {'unfreeze_layers': 1, 'strategy': 'standard'},
            'head_plus_last2': {'unfreeze_layers': 2, 'strategy': 'standard'},
            'head_plus_last3': {'unfreeze_layers': 3, 'strategy': 'standard'},
            'head_plus_last4': {'unfreeze_layers': 4, 'strategy': 'standard'},
            'head_plus_last5': {'unfreeze_layers': 5, 'strategy': 'standard'},
            'progressive': {'strategy': 'progressive'},
        }
