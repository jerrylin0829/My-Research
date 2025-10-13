"""
AttentionTargeted 遺忘器 - 注意力導向遺忘策略
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


class AttentionTargetedUnlearner(BaseUnlearner):
    """
    注意力導向遺忘：針對注意力機制進行遺忘
    
    用途與意義：
    1. 利用ViT的注意力機制特性
    2. 精確定位與遺忘數據相關的注意力模式
    3. 研究注意力與知識存儲的關係
    """
    
    def unlearn(self, retain_loader, forget_loader, 
                attention_strategy='head_specific', epochs=20, lr=1e-4,
                class_mapping=None, num_retain_classes=90):
        """
        注意力導向遺忘主方法
        
        Args:
            retain_loader: 保留數據加載器
            forget_loader: 遺忘數據加載器
            attention_strategy: 注意力策略 ('head_specific', 'layer_specific', 'pattern_based')
            epochs: 微調輪數
            lr: 學習率
            class_mapping: 類別映射
            num_retain_classes: 保留類別數
        """
        
        print(f"開始注意力導向遺忘，策略: {attention_strategy}")
        
        # 調整分類頭
        if class_mapping is not None:
            self._adjust_classification_head(num_retain_classes)
        
        if attention_strategy == 'head_specific':
            self._identify_and_modify_forget_attention_heads(forget_loader, retain_loader)
        elif attention_strategy == 'layer_specific':
            self._layer_wise_attention_modification(forget_loader, retain_loader)
        elif attention_strategy == 'pattern_based':
            self._pattern_based_attention_pruning(forget_loader, retain_loader)
        else:
            raise ValueError(f"不支援的注意力策略: {attention_strategy}")
        
        # 注意力修改後的微調
        self._fine_tune_attention(retain_loader, class_mapping, epochs, lr)
        
        return self.model
    
    def _identify_and_modify_forget_attention_heads(self, forget_loader, retain_loader):
        """識別與遺忘數據最相關的注意力頭並修改"""
        print("分析注意力頭與遺忘數據的相關性...")
        
        # 收集注意力分數
        forget_attention_scores = self._collect_attention_scores(forget_loader, "遺忘數據")
        retain_attention_scores = self._collect_attention_scores(retain_loader, "保留數據")
        
        # 找出對遺忘數據敏感但對保留數據不敏感的注意力頭
        heads_to_modify = []
        relevance_scores = {}
        
        for layer_idx in range(len(self.model.blocks)):
            for head_idx in range(self.model.blocks[layer_idx].attn.num_heads):
                key = (layer_idx, head_idx)
                forget_score = forget_attention_scores.get(key, 0)
                retain_score = retain_attention_scores.get(key, 0)
                
                # 計算遺忘相關性
                if retain_score > 1e-6:
                    forget_relevance = forget_score / retain_score
                else:
                    forget_relevance = forget_score
                
                relevance_scores[key] = forget_relevance
                
                # 如果這個頭對遺忘數據更敏感，標記為需要修改
                if forget_relevance > 1.5:  # 閾值可調
                    heads_to_modify.append(key)
        
        print(f"識別出 {len(heads_to_modify)} 個需要修改的注意力頭")
        
        # 輸出最相關的注意力頭
        sorted_heads = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        print("最相關的注意力頭:")
        for i, ((layer_idx, head_idx), score) in enumerate(sorted_heads[:10]):
            print(f"  Layer {layer_idx}, Head {head_idx}: {score:.3f}")
        
        # 修改這些注意力頭
        self._modify_attention_heads(heads_to_modify)
    
    def _collect_attention_scores(self, dataloader, data_type):
        """收集注意力分數"""
        print(f"收集{data_type}的注意力分數...")
        attention_scores = {}
        self.model.eval()
        
        def create_attention_hook(layer_idx, head_idx):
            def hook(module, input, output):
                # 對於標準的MultiHeadAttention，output可能是(output, attention_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
                        if head_idx < attn_weights.size(1):
                            head_attention = attn_weights[:, head_idx, :, :].mean().item()
                            key = (layer_idx, head_idx)
                            if key not in attention_scores:
                                attention_scores[key] = []
                            attention_scores[key].append(head_attention)
            return hook
        
        # 註冊hooks
        hooks = []
        for layer_idx, block in enumerate(self.model.blocks):
            # 假設注意力模組在block.attn中
            if hasattr(block, 'attn'):
                for head_idx in range(block.attn.num_heads):
                    hook = block.attn.register_forward_hook(create_attention_hook(layer_idx, head_idx))
                    hooks.append(hook)
        
        # 收集數據
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # 只用前5個batch
                    break
                
                inputs = batch[0].to(self.device)
                _ = self.model(inputs)
        
        # 清理hooks
        for hook in hooks:
            hook.remove()
        
        # 計算平均分數
        avg_scores = {}
        for key, scores in attention_scores.items():
            avg_scores[key] = np.mean(scores) if scores else 0
        
        return avg_scores
    
    def _modify_attention_heads(self, heads_to_modify):
        """修改指定的注意力頭"""
        print(f"修改 {len(heads_to_modify)} 個注意力頭...")
        
        for layer_idx, head_idx in heads_to_modify:
            block = self.model.blocks[layer_idx]
            
            # 獲取注意力模組
            if hasattr(block, 'attn'):
                attn = block.attn
                
                # 計算頭的維度
                head_dim = attn.embed_dim // attn.num_heads
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                # 修改該注意力頭的權重
                with torch.no_grad():
                    # 方法1: 削弱查詢權重
                    if hasattr(attn, 'q_proj'):
                        attn.q_proj.weight[start_idx:end_idx, :] *= 0.1
                    elif hasattr(attn, 'qkv'):
                        # 如果是組合的QKV權重
                        attn.qkv.weight[start_idx:end_idx, :] *= 0.1
                    
                    # 方法2: 添加噪聲
                    # noise = torch.randn_like(attn.qkv.weight[start_idx:end_idx, :]) * 0.01
                    # attn.qkv.weight[start_idx:end_idx, :] += noise
                
                print(f"修改了層 {layer_idx} 的注意力頭 {head_idx}")
    
    def _layer_wise_attention_modification(self, forget_loader, retain_loader):
        """按層修改注意力"""
        print("執行按層注意力修改...")
        
        # 分析每層的注意力模式差異
        layer_attention_diff = {}
        
        for layer_idx in range(len(self.model.blocks)):
            forget_attn = self._get_layer_attention_pattern(forget_loader, layer_idx)
            retain_attn = self._get_layer_attention_pattern(retain_loader, layer_idx)
            
            # 計算差異
            if forget_attn is not None and retain_attn is not None:
                diff = torch.abs(forget_attn - retain_attn).mean()
                layer_attention_diff[layer_idx] = diff.item()
                print(f"Layer {layer_idx} 注意力差異: {diff.item():.4f}")
        
        # 修改差異最大的幾層
        if layer_attention_diff:
            sorted_layers = sorted(layer_attention_diff.items(), key=lambda x: x[1], reverse=True)
            num_layers_to_modify = min(3, len(sorted_layers))  # 修改差異最大的3層
            
            for layer_idx, diff in sorted_layers[:num_layers_to_modify]:
                print(f"修改差異最大的層: {layer_idx} (差異: {diff:.4f})")
                
                # 對該層的所有注意力頭進行輕微修改
                block = self.model.blocks[layer_idx]
                if hasattr(block, 'attn'):
                    with torch.no_grad():
                        # 方法：添加少量噪聲來打亂注意力模式
                        if hasattr(block.attn, 'qkv'):
                            noise = torch.randn_like(block.attn.qkv.weight) * 0.005
                            block.attn.qkv.weight += noise
    
    def _get_layer_attention_pattern(self, dataloader, layer_idx):
        """獲取特定層的注意力模式"""
        attention_patterns = []
        
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attention_patterns.append(output[1].detach().cpu())
        
        # 註冊hook
        if layer_idx < len(self.model.blocks):
            handle = self.model.blocks[layer_idx].attn.register_forward_hook(hook)
        else:
            return None
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 3:  # 只用前3個batch
                    break
                inputs = batch[0].to(self.device)
                _ = self.model(inputs)
        
        handle.remove()
        
        if attention_patterns:
            return torch.cat(attention_patterns, dim=0).mean(dim=0)
        return None
    
    def _pattern_based_attention_pruning(self, forget_loader, retain_loader):
        """基於模式的注意力修剪"""
        print("執行基於模式的注意力修剪...")
        
        # 這裡實現更複雜的注意力模式分析
        # 例如：識別特定的注意力模式並選擇性地移除
        
        # 簡化版本：對所有注意力層添加少量隨機噪聲
        for layer_idx, block in enumerate(self.model.blocks):
            if hasattr(block, 'attn'):
                with torch.no_grad():
                    if hasattr(block.attn, 'qkv'):
                        # 添加小量噪聲破壞注意力模式
                        noise_scale = 0.01 * (layer_idx / len(self.model.blocks))  # 後面的層噪聲更大
                        noise = torch.randn_like(block.attn.qkv.weight) * noise_scale
                        block.attn.qkv.weight += noise
                        print(f"Layer {layer_idx}: 添加噪聲scale={noise_scale:.3f}")
    
    def _fine_tune_attention(self, retain_loader, class_mapping, epochs, lr):
        """注意力修改後的微調"""
        print(f"注意力修改後微調 {epochs} 輪...")
        
        # 只微調注意力層和分類頭
        attention_params = []
        for block in self.model.blocks:
            if hasattr(block, 'attn'):
                attention_params.extend(block.attn.parameters())
        
        # 添加分類頭參數
        if hasattr(self.model, 'head'):
            attention_params.extend(self.model.head.parameters())
        
        if not attention_params:
            print("警告：沒有注意力參數可訓練")
            return
        
        optimizer = optim.AdamW(attention_params, lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(retain_loader, desc=f"注意力微調 Epoch {epoch+1}/{epochs}")
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
                pbar.set_postfix({'loss': loss.item()})
            
            scheduler.step()
            
            # 評估
            val_acc = self._evaluate_retain(self.model, retain_loader, class_mapping)
            print(f"注意力微調 Epoch {epoch+1}: Loss={total_loss/len(retain_loader):.4f}, Val_Acc={val_acc:.2f}%")
    
    def _adjust_classification_head(self, num_retain_classes):
        """調整分類頭以適應新的類別數量"""
        if hasattr(self.model, 'head') and self.model.head.out_features != num_retain_classes:
            old_head = self.model.head
            self.model.head = nn.Linear(old_head.in_features, num_retain_classes).to(self.device)
            print(f"分類頭已調整: {old_head.out_features} -> {num_retain_classes}")
    
    def get_strategy_configs(self):
        """獲取所有可用的策略配置"""
        return {
            'head_specific': {
                'attention_strategy': 'head_specific',
                'epochs': 20,
                'lr': 1e-4
            },
            'layer_specific': {
                'attention_strategy': 'layer_specific',
                'epochs': 15,
                'lr': 5e-5
            },
            'pattern_based': {
                'attention_strategy': 'pattern_based',
                'epochs': 25,
                'lr': 1e-4
            }
        }
