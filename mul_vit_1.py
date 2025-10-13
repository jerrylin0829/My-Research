import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset
import os
import time
import json
from tqdm import tqdm
import random
from torchvision import transforms, datasets
from models.vit_LSA_classattn import VisionTransformer
import argparse
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import copy

def clone_state_dict(obj, to_cpu: bool = True):
    """
    安全深拷貝模型權重。
    允許傳入 nn.Module 或 已經是 state_dict 的 dict。
    """
    # 若是模型，先取 state_dict
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
    # 深拷貝每一個 tensor；預設搬到 CPU，節省顯存
    out = {}
    for k, v in sd.items():
        t = v.detach().clone()
        if to_cpu:
            t = t.cpu()
        out[k] = t
    return out

class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=20.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.scale = scale
    def forward(self, feats):
        f = F.normalize(feats, dim=1)
        w = F.normalize(self.W, dim=1)
        return self.scale * (f @ w.t())

class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n].data)

def compute_class_weights_from_subset(subset, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.long)
    ds = subset.dataset
    for idx in subset.indices:
        _, y = ds[idx]
        counts[y] += 1
    counts.clamp_(min=1)
    inv = 1.0 / counts.float()
    return (inv / inv.mean()).float()
         
class MachineUnlearning:
    """
    機器遺忘(Machine Unlearning)框架 - 實現和評估模型遺忘功能
    包含多種評估指標: MIA(成員推斷攻擊)、GS(黃金標準)等
    """
    def __init__(self, 
                 model_class=VisionTransformer, 
                 model_args=None, 
                 device='cuda',
                 batch_size=128,
                 output_dir='./unlearning_output',
                 log_dir: Optional[str] = None,
                 args=None, 
                 ):
        """
        初始化機器遺忘框架
        
        Args:
            model_class: 模型類
            model_args: 模型初始化參數
            device: 計算設備
            output_dir: 輸出目錄
        """
        self.model_class = model_class
        self.model_args = model_args or {}
        self.embed_dim = self.model_args.get('embed_dim', 300)
        self.num_classes = self.model_args.get('num_classes', None)
        self.batch_size = batch_size
        self.device = device
        self.output_dir = output_dir
        self.original_labels_test = None
        
        # 讀取（若有）CLI 旗標，給 GS/評估用
        self.gs_use_ema = bool(getattr(args, "gs_use_ema", False))
        self.gs_ema_decay = float(getattr(args, "gs_ema_decay", 0.999))
        self.gs_use_class_balance = bool(getattr(args, "gs_use_class_balance", False))
        self.gs_warmup_epochs = int(getattr(args, "gs_warmup_epochs", 5))

        # 10/4 00:40 更新這四行
        # self.use_cosine_head = getattr(self, "use_cosine_head", True)
        # self.cosine_scale = getattr(self, "cosine_scale", 20.0)
        # if self.use_cosine_head:
        #     self.head = CosineClassifier(in_dim=self.embed_dim, num_classes=self.num_classes, scale=self.cosine_scale)

        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化（或使用外部指定）TensorBoard 目錄
        tb_dir = log_dir if log_dir is not None else os.path.join(self.output_dir, "logs")
        os.makedirs(tb_dir, exist_ok=True)

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=tb_dir)
        
        # 初始化模型
        self.original_model = None  # 原始模型（遺忘前）
        self.unlearned_model = None  # 遺忘後的模型
        self.retrained_model = None  # 重新訓練的模型（黃金標準）
        self.random_model = None  # 隨機模型（未經訓練）

        # 跟踪實驗結果
        self.results = {
            'unlearning_time': 0,
            'retraining_time': 0,
            'original_metrics': {},
            'unlearned_metrics': {},
            'retrained_metrics': {},
            'mia_results': {},
            'gs_comparison': {}
        }
        
        # 設置隨機種子
        self.set_seed(42)
    
    @staticmethod
    def set_seed(seed):
        """設置隨機種子以確保可重現性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_original_model(self, model_path):
        """載入預訓練的原始模型"""
        self.original_model = self.model_class(**self.model_args).to(self.device)
        self.original_model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"原始模型已載入: {model_path}")
        
        # 複製一份作為將要進行遺忘的模型
        self.unlearned_model = self.model_class(**self.model_args).to(self.device)
        self.unlearned_model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    # 初始化隨機模型，用於 zrf 指標
    def init_random_model(self):
        """
        初始化隨機模型（不訓練）—用於正確的 ZRF 計算
        """
        self.random_model = self.model_class(**self.model_args).to(self.device)
        self.random_model.eval()
        return self.random_model
    
    def prepare_data(self, 
                     dataset_name='CIFAR100', 
                     data_path='./data',
                     forget_classes=None,
                     forget_indices=None
        ):
        """
        準備數據集，分割為記住和遺忘部分
        
        Args:
            dataset_name: 數據集名稱
            data_path: 數據集路徑
            forget_classes: 要遺忘的類別集合(與forget_indices二選一)
            forget_indices: 要遺忘的樣本索引(與forget_classes二選一)
        """
        # 數據轉換
        mean = [0.5071, 0.4867, 0.4408] 
        std = [0.2675, 0.2565, 0.2761]
        
        # 增強數據轉換 - 用於訓練
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.TrivialAugmentWide(),
            # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.3, 0.1)], 
            p=0.5
        ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 先前的版本
        # transfrom_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # 加載數據集
        if dataset_name.upper() == 'CIFAR100':
            train_set = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # 分割遺忘集和保留集
        if forget_classes is not None:
            # 先確保可索引（避免 set 造成後續 tensor indexing 出錯）
            if isinstance(forget_classes, set):
                forget_classes = sorted(list(forget_classes))
            elif isinstance(forget_classes, np.ndarray):
                forget_classes = sorted(list(forget_classes.tolist()))
            else:
                forget_classes = sorted(list(forget_classes))

            # 按類別分割
            retain_indices = [i for i, (_, label) in enumerate(train_set) if label not in forget_classes]
            forget_indices = [i for i, (_, label) in enumerate(train_set) if label in forget_classes]
            
            # 測試集也需要相應分割
            retain_test_indices = [i for i, (_, label) in enumerate(test_set) if label not in forget_classes]
            forget_test_indices = [i for i, (_, label) in enumerate(test_set) if label in forget_classes]
            
            self.num_retain_classes = len(set(label for _, label in train_set) - set(forget_classes))
            self.total_classes = len(set(label for _, label in train_set))
            self.forget_classes = forget_classes
            
            print(f"遺忘類別: {forget_classes}")
            print(f"保留類別數: {self.num_retain_classes}, 遺忘類別數: {len(forget_classes)}")
        elif forget_indices is not None:
            # 直接使用提供的索引
            retain_indices = [i for i in range(len(train_set)) if i not in forget_indices]
            
            # 假設測試集需要按原始比例分割
            retain_test_indices = list(range(len(test_set)))
            forget_test_indices = []
            
            self.num_retain_classes = self.total_classes = len(set(label for _, label in train_set))
            self.forget_classes = None
            
            print(f"遺忘樣本數: {len(forget_indices)}, 保留樣本數: {len(retain_indices)}")
        else:
            raise ValueError("必須指定forget_classes或forget_indices")
        
        # 創建數據子集
        self.retain_train_set = Subset(train_set, retain_indices)
        self.forget_train_set = Subset(train_set, forget_indices)
        self.retain_test_set = Subset(test_set, retain_test_indices)
        self.forget_test_set = Subset(test_set, forget_test_indices)

        self.class_mapping = None   # 不使用類別映射，保持原始100類
        
        # 創建數據加載器
        num_workers = 8
        pin = True
        persist = num_workers > 0

        self.retain_train_loader = DataLoader(self.retain_train_set, batch_size=self.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin, persistent_workers=persist)
        self.forget_train_loader = DataLoader(self.forget_train_set, batch_size=self.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin, persistent_workers=persist)
        self.retain_test_loader = DataLoader(self.retain_test_set, batch_size=self.batch_size, shuffle=False, 
                                             num_workers=num_workers, pin_memory=pin, persistent_workers=persist)
        self.forget_test_loader = DataLoader(self.forget_test_set, batch_size=self.batch_size, shuffle=False, 
                                             num_workers=num_workers, pin_memory=pin, persistent_workers=persist)

        print(f"數據準備完成:")
        print(f"  保留訓練集: {len(self.retain_train_set)} 樣本")
        print(f"  遺忘訓練集: {len(self.forget_train_set)} 樣本")
        print(f"  保留測試集: {len(self.retain_test_set)} 樣本")
        print(f"  遺忘測試集: {len(self.forget_test_set)} 樣本")
    
    def get_scheduler(self, optimizer, scheduler_type, epochs, train_loader=None, min_lr=1e-6, onecycle_max_lr=None):
        """
        根據指定的調度器類型返回學習率調度器
        
        Args:
            optimizer: 優化器
            scheduler_type: 調度器類型 ('cosine', 'step', 'plateau', 'onecycle')
            epochs: 訓練輪數
            train_loader: 訓練數據加載器(OneCycleLR需要)
            min_lr: 最小學習率
        
        Returns:
            lr_scheduler: 學習率調度器
        """
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=min_lr
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs//3, gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=min_lr
            )
        elif scheduler_type == 'onecycle':
            max_lr = onecycle_max_lr if onecycle_max_lr is not None else optimizer.param_groups[0]['lr'] * 10.0
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
            )
        elif scheduler_type == 'cosine_warmup':
            # 先 warmup，再 cosine，到最後一步回到 eta_min / base_lr 的比例
            assert train_loader is not None, "cosine_warmup 需要 train_loader 來計算總步數"
            total_steps = max(1, epochs * len(train_loader))
            warmup_epochs = int(getattr(self, "gs_warmup_epochs", 5))
            warmup_steps = max(1, warmup_epochs * len(train_loader))
            base_lr = optimizer.param_groups[0]['lr']
            floor = float(min_lr) / float(base_lr)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return floor + (1.0 - floor) * cosine

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def _step_scheduler_per_batch(self, scheduler_type, scheduler):
        return scheduler_type in {'onecycle', 'cosine_warmup'}

    def _step_scheduler_per_epoch(self, scheduler_type):
        return scheduler_type in {'cosine', 'step'}

    def unlearn_by_retraining_head(self, epochs=10, lr=5e-5, lr_scheduler_type='cosine', min_lr=1e-6):
        """
        通過重新訓練頭部實現機器遺忘
        - 凍結所有層，僅重新訓練分類頭
        - 只使用保留數據集進行訓練
        
        Args:
            epochs: 訓練輪數
            lr: 學習率
            lr_scheduler_type: 學習率調度器類型
            min_lr: 最小學習率
        """
        assert self.unlearned_model is not None, "請先加載原始模型"
        
        print("開始頭部重訓練遺忘...")
        start_time = time.time()
        
        # 凍結所有參數
        for param in self.unlearned_model.parameters():
            param.requires_grad = False
        
        # 設置頭部參數為可訓練
        for param in self.unlearned_model.head.parameters():
            param.requires_grad = True
        
        # 設置優化器
        optimizer = optim.AdamW(self.unlearned_model.head.parameters(), lr=lr, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 設置學習率調度器
        scheduler = self.get_scheduler(optimizer, lr_scheduler_type, epochs, self.retain_train_loader, min_lr,
                                   onecycle_max_lr=lr*10 if lr_scheduler_type=='onecycle' else None)

        
        # 訓練循環
        best_acc = 0
        best_state = None
        
        for epoch in range(epochs):
            self.unlearned_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(self.retain_train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                new_labels = labels
                
                optimizer.zero_grad()
                outputs = self.unlearned_model(inputs)
                ce_loss = criterion(outputs, labels)

                # === 熵正則 ===
                probs = F.softmax(outputs, dim=1)
                entropy_reg = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()

                # === 總 loss ===
                lambda_entropy = 1e-3  # entropy 正則化權重, 建議 1e-3 ~ 5e-3
                loss = ce_loss + lambda_entropy * entropy_reg

                loss.backward()
                optimizer.step()
                
                # if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                #     scheduler.step()
                if self._step_scheduler_per_batch(lr_scheduler_type, scheduler):
                    scheduler.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += new_labels.size(0)
                correct += predicted.eq(new_labels).sum().item()
            
            train_loss = running_loss / len(self.retain_train_loader)
            train_acc = 100 * correct / total
            
            # 評估保留集性能
            retain_acc = self._evaluate_retain(self.unlearned_model)
            
            # 🔧 Cosine/Step 在 epoch 結尾 step；Plateau 用 metric
            if self._step_scheduler_per_epoch(lr_scheduler_type):
                scheduler.step()
            
            # 更新學習率(如果使用plateau)
            if lr_scheduler_type == 'plateau':
                scheduler.step(retain_acc)
            
            # 記錄到TensorBoard
            self.writer.add_scalar('UnlearnHead/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('UnlearnHead/Train_Acc', train_acc, epoch)
            self.writer.add_scalar('UnlearnHead/Retain_Test_Acc', retain_acc, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar('UnlearnHead/Learning_Rate', current_lr, epoch)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Retain Test Acc: {retain_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if retain_acc > best_acc:
                best_acc = retain_acc
                best_state = clone_state_dict(self.unlearned_model)

        # 恢復最佳模型
        self.unlearned_model.load_state_dict(best_state)
        
        end_time = time.time()
        self.results['unlearning_time'] = end_time - start_time
        print(f"遺忘完成! 耗時: {self.results['unlearning_time']:.2f} 秒")
        
        # 保存遺忘後的模型
        torch.save(self.unlearned_model.state_dict(), 
                  f"{self.output_dir}/unlearned_head_retrain_model.pth")
    
    def unlearn_by_negative_gradient(self, epochs=10, lr=1e-5, retain_epochs=5, 
                                   lr_scheduler_type='cosine', min_lr=1e-6):
        """
        通過負梯度更新實現機器遺忘
        1. 對遺忘數據執行梯度上升(使模型"忘記")
        2. 對保留數據執行標準梯度下降(保持性能)
        
        Args:
            epochs: 負梯度訓練輪數
            lr: 學習率
            retain_epochs: 保留數據微調輪數
            lr_scheduler_type: 學習率調度器類型
            min_lr: 最小學習率
        """
        assert self.unlearned_model is not None, "請先加載原始模型"
        
        print("開始負梯度遺忘...")
        start_time = time.time()
        
        # 解凍所有參數
        for param in self.unlearned_model.parameters():
            param.requires_grad = True
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.unlearned_model.parameters(), lr=lr, weight_decay=0.05)
        
        # 設置學習率調度器
        scheduler = self.get_scheduler(
            optimizer, 
            lr_scheduler_type, 
            epochs + retain_epochs, 
            self.retain_train_loader, 
            min_lr,
            onecycle_max_lr=lr*10 if lr_scheduler_type == 'onecycle' else None
        )
        
        # 階段1: 負梯度更新(梯度上升)
        for epoch in range(epochs):
            self.unlearned_model.train()
            running_loss = 0.0
            
            for inputs, labels in tqdm(self.forget_train_loader, desc=f"忘記階段 {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.unlearned_model(inputs)
                ce_loss = criterion(outputs, labels)

                # === 熵正則 ===
                probs = F.softmax(outputs, dim=1)
                entropy_reg = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()

                # === 總 loss ===
                lambda_entropy = 1e-3  # entropy 正則化權重, 建議 1e-3 ~ 5e-3
                loss = ce_loss + lambda_entropy * entropy_reg
                
                # 負梯度更新 - 梯度上升使模型"忘記"
                (-loss).backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            forget_loss = running_loss / len(self.forget_train_loader)
            
            # 評估保留數據性能
            retain_acc = self._evaluate_retain(self.unlearned_model)
            
            # 更新學習率
            if lr_scheduler_type != 'plateau':
                scheduler.step()
            else:
                scheduler.step(retain_acc)
            
            # 記錄到TensorBoard
            self.writer.add_scalar('NegGrad/Forget_Loss', forget_loss, epoch)
            self.writer.add_scalar('NegGrad/Retain_Acc', retain_acc, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar('NegGrad/Learning_Rate', current_lr, epoch)
            
            print(f"忘記階段 Epoch {epoch+1}/{epochs}, Loss: {forget_loss:.4f}")
            print(f"  Retain Test Acc: {retain_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # 階段2: 保留數據微調
        print("開始保留數據微調...")
        for epoch in range(retain_epochs):
            self.unlearned_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(self.retain_train_loader, desc=f"保留階段 {epoch+1}/{retain_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.unlearned_model(inputs)
                ce_loss = criterion(outputs, labels)

                # === 熵正則 ===
                probs = F.softmax(outputs, dim=1)
                entropy_reg = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()

                # === 總 loss ===
                lambda_entropy = 1e-3  # entropy 正則化權重, 建議 1e-3 ~ 5e-3
                loss = ce_loss + lambda_entropy * entropy_reg

                # 標準梯度下降
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            retain_loss = running_loss / len(self.retain_train_loader)
            retain_acc = 100 * correct / total
            
            # 評估保留數據性能
            retain_test_acc = self._evaluate_retain(self.unlearned_model)
            
            # 更新學習率
            if lr_scheduler_type != 'plateau':
                scheduler.step()
            else:
                scheduler.step(retain_test_acc)
            
            # 記錄到TensorBoard
            self.writer.add_scalar('NegGrad/Retain_Loss', retain_loss, epochs + epoch)
            self.writer.add_scalar('NegGrad/Retain_Train_Acc', retain_acc, epochs + epoch)
            self.writer.add_scalar('NegGrad/Retain_Test_Acc', retain_test_acc, epochs + epoch)
            current_lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar('NegGrad/Learning_Rate', current_lr, epochs + epoch)
            
            print(f"保留階段 Epoch {epoch+1}/{retain_epochs}, Loss: {retain_loss:.4f}, Acc: {retain_acc:.2f}%")
            print(f"  Retain Test Acc: {retain_test_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
        
        end_time = time.time()
        self.results['unlearning_time'] = end_time - start_time
        print(f"遺忘完成! 耗時: {self.results['unlearning_time']:.2f} 秒")
        
        # 保存遺忘後的模型
        torch.save(self.unlearned_model.state_dict(), 
                  f"{self.output_dir}/unlearned_neg_grad_model.pth")
    
    # === Mixup / CutMix ===
    def _apply_mix_augment(self, inputs, targets, alpha=0.2, use_cutmix=False):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size, device=inputs.device)

        # 🛠️ 新增最小λ限制，避免過度混合
        lam = max(lam, 0.8)  # 確保至少80%是原始圖像

        if use_cutmix:
            # cutmix：隨機矩形區塊交換
            H, W = inputs.size(2), inputs.size(3)
            cx, cy = np.random.randint(W), np.random.randint(H)
            w = int(W * np.sqrt(1 - lam))
            h = int(H * np.sqrt(1 - lam))
            x0 = np.clip(cx - w // 2, 0, W); x1 = np.clip(cx + w // 2, 0, W)
            y0 = np.clip(cy - h // 2, 0, H); y1 = np.clip(cy + h // 2, 0, H)
            inputs[:, :, y0:y1, x0:x1] = inputs[index, :, y0:y1, x0:x1]
            lam = 1 - ((x1 - x0) * (y1 - y0) / (W * H))
            targets_a, targets_b = targets, targets[index]
            return inputs, targets_a, targets_b, lam
        else:
            # mixup：整張線性混合
            mixed = lam * inputs + (1 - lam) * inputs[index]
            targets_a, targets_b = targets, targets[index]
            return mixed, targets_a, targets_b, lam
        
    def train_gold_standard(self, epochs=250, 
                            lr=1e-3, 
                            lr_scheduler_type='cosine', 
                            min_lr=1e-6, 
                            weight_decay=0.05,
                            use_mixup=False,
                            use_cutmix=False,
                            mix_alpha=0.2,
                            use_logit_penalty=False,
                            experiment_writer=None):
        """
        訓練黃金標準（GS）模型：僅用 retain set，自零開始訓練，作為對齊與比較基準。
        - 支援 Class-balanced CE（僅對 retain 訓練集）
        - 支援 EMA（每 step 更新；驗證前套用；驗證後還原；最佳以 EMA 權重保存）
        - 支援 Cosine + Warmup（lr_scheduler_type='cosine_warmup'；每 batch step）
        - Mixup/CutMix 使用軟準確率；否則用標準準確率
        """
        print("開始訓練黃金標準模型...")
        start_time = time.time()
        if not hasattr(self, "results"):
            self.results = {}

        # 1) 建立全新模型（GS）並移動到 device
        self.retrained_model = self.model_class(**self.model_args).to(self.device)

        # 2) Optimizer / Criterion（Class-balanced CE 僅對 retain 訓練集）
        optimizer = optim.AdamW(self.retrained_model.parameters(), lr=lr, weight_decay=weight_decay)

        use_class_balance = bool(getattr(self, "gs_use_class_balance", False))
        if use_class_balance:
            cw = compute_class_weights_from_subset(self.retain_train_set, self.num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 3) Scheduler（支援 cosine_warmup / onecycle 需要 train_loader）
        scheduler = self.get_scheduler(
            optimizer, lr_scheduler_type, epochs, self.retain_train_loader, min_lr,
            onecycle_max_lr=lr*10 if lr_scheduler_type == 'onecycle' else None
        )

        # 4) EMA
        use_ema = bool(getattr(self, "gs_use_ema", False))
        ema_decay = float(getattr(self, "gs_ema_decay", 0.999))
        ema_helper = EMAHelper(self.retrained_model, decay=ema_decay) if use_ema else None

        # 5) 訓練控制
        best_acc = 0.0
        best_state = None
        patience = 20
        no_improve = 0

        for epoch in range(epochs):
            self.retrained_model.train()
            running_loss = 0.0
            total = 0
            correct_sum = 0.0  # 累積 batch-wise 準確率（可能是軟準確率）
            
            for inputs, labels in tqdm(self.retain_train_loader, desc=f"GS Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # === 前向：是否使用 Mixup/CutMix ===
                if use_mixup or use_cutmix:
                    inputs, ta, tb, lam = self._apply_mix_augment(inputs, labels, alpha=mix_alpha, use_cutmix=use_cutmix)
                    outputs = self.retrained_model(inputs)
                    ce_loss = lam * criterion(outputs, ta) + (1.0 - lam) * criterion(outputs, tb)

                    with torch.no_grad():
                        pred = outputs.argmax(dim=1)
                        acc_a = (pred == ta).float().mean()
                        acc_b = (pred == tb).float().mean()
                        batch_acc = (lam * acc_a + (1.0 - lam) * acc_b).item()
                else:
                    outputs = self.retrained_model(inputs)
                    ce_loss = criterion(outputs, labels)
                    with torch.no_grad():
                        pred = outputs.argmax(dim=1)
                        batch_acc = (pred == labels).float().mean().item()

                # === 正則：熵正則（偏小，避免過度自信）＋（可選）logit 範數懲罰 ===
                probs = F.softmax(outputs, dim=1)
                entropy_reg = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()
                lambda_entropy = 1e-3

                if use_logit_penalty:
                    logit_norm = outputs.norm(p=2, dim=1).mean()
                    lambda_logit = 1e-4
                    loss = ce_loss + lambda_entropy * entropy_reg + lambda_logit * logit_norm
                else:
                    loss = ce_loss + lambda_entropy * entropy_reg

                # === 反向傳播 ===
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.retrained_model.parameters(), max_norm=1.0)
                optimizer.step()

                # === EMA：每步更新 ===
                if ema_helper is not None:
                    ema_helper.update(self.retrained_model)

                # === Scheduler：batch-step 的類型（onecycle / cosine_warmup）在這裡 step ===
                if self._step_scheduler_per_batch(lr_scheduler_type, scheduler):
                    scheduler.step()

                running_loss += loss.item()
                total += labels.size(0)
                correct_sum += batch_acc * labels.size(0)

            # === Epoch 統計 ===
            train_loss = running_loss / max(1, len(self.retain_train_loader))
            train_acc = 100.0 * correct_sum / max(1, total)

            # === 驗證前：若有 EMA，先套用到模型；驗證後還原 ===
            if ema_helper is not None:
                current_state = {k: v.detach().clone() for k, v in self.retrained_model.state_dict().items()}
                ema_helper.apply_to(self.retrained_model)

            retain_acc = self._evaluate_retain(self.retrained_model)

            # 以當前生效的權重（若有 EMA 即為 EMA 權重）判斷最佳，並保存「EMA 權重」版本
            if retain_acc > best_acc:
                best_acc = retain_acc
                best_state = {k: v.detach().clone().cpu() for k, v in self.retrained_model.state_dict().items()}
                no_improve = 0
                print(f"  [New Best] Acc: {retain_acc:.2f}%")
            else:
                no_improve += 1

            # 還原回即時權重（以便下一個 epoch 繼續訓練）
            if ema_helper is not None:
                self.retrained_model.load_state_dict(current_state)

            # === Scheduler：epoch-step 的類型（cosine/step）或 plateau 在這裡 step ===
            if self._step_scheduler_per_epoch(lr_scheduler_type):
                scheduler.step()
            elif lr_scheduler_type == 'plateau':
                scheduler.step(retain_acc)

            # === TensorBoard 記錄 ===
            writer = experiment_writer if experiment_writer is not None else getattr(self, "writer", None)
            if writer is not None:
                writer.add_scalar('GoldStandard_Training/Train_Loss', train_loss, epoch)
                writer.add_scalar('GoldStandard_Training/Train_Acc', train_acc, epoch)
                writer.add_scalar('GoldStandard_Training/Retain_Test_Acc', retain_acc, epoch)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('GoldStandard_Training/Learning_Rate', current_lr, epoch)

            print(f"GS Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Retain Test Acc (EMA eval={'on' if ema_helper else 'off'}): {retain_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # 早停
            if no_improve >= patience:
                print(f"早停: {patience} 輪沒有改善")
                break

        # === 載入最佳（EMA 權重版本） ===
        if best_state is not None:
            self.retrained_model.load_state_dict(best_state)

        end_time = time.time()
        self.results['retraining_time'] = end_time - start_time
        print(f"黃金標準模型訓練完成! 耗時: {self.results['retraining_time']:.2f} 秒")

        # 保存 GS 權重
        try:
            torch.save(self.retrained_model.state_dict(), f"{self.output_dir}/gold_standard_model.pth")
        except Exception as e:
            print(f"[WARN] 保存 GS 權重失敗：{e}")
    
    def _evaluate_retain(self, model):
        """評估模型在保留集上的性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.retain_test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
           
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100 * correct / total
    
    def _evaluate_forget(self, model):
        """
        評估模型在遺忘集上的性能
        """
        if len(self.forget_test_loader.dataset) == 0:
            return None

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.forget_test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                _, pred = outputs.max(1)

                correct += pred.eq(labels).sum().item()
                total   += labels.size(0)
        
        return 100 * correct / total if total else None
    
    def _forget_proxy_metrics(self, model):
        """
        類別遺忘時，用不依賴真實標籤的代理指標衡量「忘記程度」：
        - MSP (Max Softmax Probability) 越低越好
        - Entropy 越高越好
        """
        if len(self.forget_test_loader.dataset) == 0:
            return None
        model.eval()
        
        msp_vals, ent_vals = [], []
        with torch.no_grad():
            for x, _ in self.forget_test_loader:
                x = x.to(self.device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                msp = probs.max(dim=1).values
                ent = -(probs * (probs + 1e-12).log()).sum(dim=1)
                msp_vals.append(msp.cpu())
                ent_vals.append(ent.cpu())
        if not msp_vals:
            return None
        msp_mean = torch.cat(msp_vals).mean().item()
        ent_mean = torch.cat(ent_vals).mean().item()
        return {'msp': msp_mean, 'entropy': ent_mean}

    def evaluate_all_models(self):
        """評估所有模型的性能"""
        print("\n===== 評估所有模型 =====")
        
        # 評估原始模型
        if self.original_model is not None:
            orig_retain_acc = self._evaluate_retain(self.original_model)
            orig_forget_acc = self._evaluate_forget(self.original_model)
            proxy = self._forget_proxy_metrics(self.original_model)
            self.results['original_metrics'] = {
                'retain_acc': orig_retain_acc,
                'forget_acc': orig_forget_acc,
                'forget_proxy': proxy
            }
            print(f"原始模型 - 保留集: {orig_retain_acc:.2f}%, 遺忘集: {orig_forget_acc:.2f}%")
            if proxy:
                print(f"  Forget MSP: {proxy['msp']:.4f}, Entropy: {proxy['entropy']:.4f}")
        
        # 評估遺忘後的模型
        if self.unlearned_model is not None:
            unl_retain_acc = self._evaluate_retain(self.unlearned_model)
            unl_forget_acc = self._evaluate_forget(self.unlearned_model)
            proxy = self._forget_proxy_metrics(self.unlearned_model)
            self.results['unlearned_metrics'] = {
                'retain_acc': unl_retain_acc,
                'forget_acc': unl_forget_acc,
                'forget_proxy': proxy
            }
            print(f"遺忘模型 - 保留集: {unl_retain_acc:.2f}%, 遺忘集: {unl_forget_acc if unl_forget_acc is not None else 'N/A'}%")
            if proxy:
                print(f"  Forget MSP: {proxy['msp']:.4f}, Entropy: {proxy['entropy']:.4f}")
        # 黃金標準
        if self.retrained_model is not None:
            gs_retain_acc = self._evaluate_retain(self.retrained_model)
            gs_forget_acc = self._evaluate_forget(self.retrained_model)
            proxy = self._forget_proxy_metrics(self.retrained_model)
            self.results['retrained_metrics'] = {
                'retain_acc': gs_retain_acc,
                'forget_acc': gs_forget_acc,
                'forget_proxy': proxy
            }
            print(f"黃金標準 - 保留集: {gs_retain_acc:.2f}%, 遺忘集: {gs_forget_acc if gs_forget_acc is not None else 'N/A'}%")
            if proxy:
                print(f"  Forget MSP: {proxy['msp']:.4f}, Entropy: {proxy['entropy']:.4f}")
    
    # 1) 檢查 head 維度、是否 100 類  # // NEW
    def check_head_integrity(self, model):
        head = getattr(model, 'head', None)
        ok = True
        if head is None:
            print("[Head Check] 模型沒有 head 層屬性")
            ok = False
        else:
            out_features = getattr(head, 'out_features', None)
            in_features  = getattr(head, 'in_features', None)
            print(f"[Head Check] head.in_features={in_features}, head.out_features={out_features}, num_classes(meta)={getattr(self,'num_classes',None)}")
            if out_features is not None and hasattr(self, 'num_classes'):
                if out_features != self.num_classes:
                    print("[Head Check][警告] head 輸出維度與 num_classes 不一致！")
                    ok = False
        return ok

    # 2) 檢查 head.weight 對各類的整列是否全 0（zero-lock 的直接指標）  # // NEW
    def check_head_zero_columns(self, model, forget_classes):
        head = getattr(model, 'head', None)
        if head is None or not hasattr(head, 'weight'):
            print("[Head Zeros] 無法檢查（沒有 head.weight）")
            return None
        
        W = head.weight.detach().cpu()  # [C, D]
        col_zero = (W.abs().sum(dim=1) == 0)  # 每一列是否全 0（對應一個類別）
        num_zero = int(col_zero.sum().item())
        num_total = W.size(0)
        print(f"[Head Zeros] 全 0 列：{num_zero}/{num_total} ({100.0*num_zero/num_total:.2f}%)")

        if forget_classes is not None:
            # 強健處理：把任何型別轉成可索引的 LongTensor
            if isinstance(forget_classes, (set, list, tuple, np.ndarray)):
                idx = torch.tensor(sorted(list(forget_classes)), dtype=torch.long)
            elif torch.is_tensor(forget_classes):
                idx = forget_classes.to(dtype=torch.long, device=col_zero.device).cpu()
            else:
                raise TypeError(f"forget_classes 型別不支援索引: {type(forget_classes)}")

            # col_zero 是 CPU tensor，確保 idx 在 CPU
            idx = idx.cpu()
            forget_zeros = int(col_zero.index_select(dim=0, index=idx).sum().item())
            print(f"[Head Zeros] 忘記類中全 0 列：{forget_zeros}/{len(idx)} ({100.0*forget_zeros/max(len(idx),1):.2f}%)")

        return col_zero

    # 3) 忘記測試集：預測落在「忘記類 vs 保留類」的比例  # // NEW
    def debug_forget_prediction_stats(self, model):
        assert hasattr(self, 'forget_classes') and self.forget_classes is not None, "沒有 forget_classes 可用"
        fc = set(self.forget_classes)
        model.eval()
        total, pred_in_forget, pred_in_retain = 0, 0, 0
        with torch.no_grad():
            for x, _ in self.forget_test_loader:
                x = x.to(self.device)
                pred = model(x).argmax(dim=1).cpu().tolist()
                for p in pred:
                    if p in fc: pred_in_forget += 1
                    else:       pred_in_retain += 1
                total += len(pred)
        r_forget = pred_in_forget / max(total,1)
        r_retain = pred_in_retain / max(total,1)
        print(f"[Forget Pred Stats] total={total}, pred_in_forget={pred_in_forget} ({r_forget:.3f}), pred_in_retain={pred_in_retain} ({r_retain:.3f})")
        return r_forget, r_retain

    # 4) 忘記測試集：真實類（屬於忘記）的 logit 直方圖（數據化 + 圖檔）  # // NEW
    def plot_true_class_logit_hist(self, model, tag, out_dir):
        import os, torch
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        vals = []
        with torch.no_grad():
            for x, y in self.forget_test_loader:
                x = x.to(self.device); y = y.to(self.device)
                z = model(x)  # (B, C)
                true_logit = z.gather(1, y.view(-1,1)).squeeze(1)
                vals.append(true_logit.cpu())
        if not vals:
            print(f"[{tag}] 無忘記測試樣本；跳過繪圖")
            return
        v = torch.cat(vals).numpy()
        plt.figure()
        plt.hist(v, bins=50)
        plt.title(f"True-class logits on FORGET ({tag})")
        plt.xlabel("logit (true class)"); plt.ylabel("count")
        fp = os.path.join(out_dir, f"{tag}_forget_trueclass_logit_hist.png")
        plt.savefig(fp, dpi=180); plt.close()
        print(f"[Save] {fp}")

    # 5) 忘記測試集：類別分佈（預測出現次數前 10 名）  # // NEW
    def top_predicted_classes_on_forget(self, model, topk=10):
        from collections import Counter
        model.eval()
        ctr = Counter()
        with torch.no_grad():
            for x, _ in self.forget_test_loader:
                x = x.to(self.device)
                pred = model(x).argmax(dim=1).cpu().tolist()
                ctr.update(pred)
        total = sum(ctr.values())
        top = ctr.most_common(topk)
        print("[Forget Top Pred Classes] (class_id, count, ratio)")
        for cid, cnt in top:
            print(f"  {cid:3d}  {cnt:6d}  {cnt/max(total,1):.3f}")
        return ctr

    def plot_forget_distributions(self, model, tag: str, out_dir: str, exp_id: str = None):
        """
        對 forget test set 畫 MSP / Entropy / Logit L2 範數 的直方圖
        存成三張圖：{tag}_msp.png, {tag}_entropy.png, {tag}_logit_norm.png
        """
        if len(self.forget_test_loader.dataset) == 0:
            return None
        model.eval()
        msp_list=[]; ent_list=[]; norm_list=[]

        with torch.no_grad():
            for x, _ in self.forget_test_loader:
                x = x.to(self.device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                msp = probs.max(dim=1).values
                ent = -(probs * (probs + 1e-12).log()).sum(dim=1)
                ln = logits.norm(p=2, dim=1)

                msp_list.append(msp.cpu()); ent_list.append(ent.cpu()); norm_list.append(ln.cpu())

        os.makedirs(out_dir, exist_ok=True)

        def _save_hist(arr, title, fname, xlabel=None):
            arr = torch.cat(arr).numpy()
            plt.figure(figsize=(5,4))
            plt.hist(arr, bins=40)
            plt.title(title) 
            plt.xlabel(xlabel)
            plt.ylabel("count (samples)")
            plt.tight_layout()

            final_fname = f"{exp_id}_{fname}" if exp_id else fname
            path = os.path.join(out_dir, final_fname)
            plt.savefig(path); plt.close()
            print(f"[Distributions] saved: {path}")

        _save_hist(msp_list, f"{tag} MSP ({exp_id or ''})", f"{tag}_msp.png", xlabel="max softmax probability")
        _save_hist(ent_list, f"{tag} Entropy ({exp_id or ''})", f"{tag}_entropy.png", xlabel="entropy (−∑p log p)")
        _save_hist(norm_list, f"{tag} Logit L2 Norm ({exp_id or ''})", f"{tag}_logit_norm.png", xlabel="‖logits‖₂")

    def calculate_metrics(self):
        """計算機器遺忘指標"""
        if not self.results['original_metrics'] or not self.results['unlearned_metrics']:
            print("請先評估模型性能")
            return
        
        # 提取指標
        orig_retain = self.results['original_metrics']['retain_acc']
        orig_forget = self.results['original_metrics']['forget_acc']
        unl_retain = self.results['unlearned_metrics']['retain_acc']
        unl_forget = self.results['unlearned_metrics']['forget_acc'] or 0
        
        # 計算遺忘效果 (1 - 遺忘集準確率下降比例)
        if orig_forget > 0:
            forget_effect = (orig_forget - unl_forget) / orig_forget
        else:
            forget_effect = 1.0
        
        # 計算保留效果 (保留集準確率相對變化)
        retain_effect = (unl_retain - orig_retain) / orig_retain
        
        # 計算綜合指標
        if 'retrained_metrics' in self.results and self.results['retrained_metrics']:
            gs_retain = self.results['retrained_metrics']['retain_acc']
            
            # GS近似度 - 與黃金標準的接近程度
            gs_similarity = unl_retain / gs_retain if gs_retain > 0 else 0
            
            self.results['gs_comparison'] = {
                'gs_similarity': gs_similarity,
                'unl_vs_gs_diff': unl_retain - gs_retain
            }
        
        # 保存結果
        self.results['forget_effect'] = forget_effect
        self.results['retain_effect'] = retain_effect
        self.results['MU_score'] = forget_effect * (1 + retain_effect)  # 綜合機器遺忘分數
        
        print("\n===== 機器遺忘指標 =====")
        print(f"遺忘效果: {forget_effect:.4f} (1.0 = 完全遺忘)")
        print(f"保留效果: {retain_effect:.4f} (>0 = 知識保留得更好)")
        print(f"MU綜合分數: {self.results['MU_score']:.4f} (越高越好)")
        
        if 'gs_comparison' in self.results:
            print(f"黃金標準近似度: {self.results['gs_comparison']['gs_similarity']:.4f}")
            print(f"與黃金標準差異: {self.results['gs_comparison']['unl_vs_gs_diff']:.2f}%")
        
        return self.results

    def save_complete_report(self, filepath=None):
        """
        保存詳細的實驗報告，包括所有參數設定和結果
        
        Args:
            filepath: 報告存儲路徑，如果為None則使用默認路徑
        """
        if filepath is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f"{self.output_dir}/unlearning_report_{timestamp}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 1. 報告標題
            f.write("="*80 + "\n")
            f.write(f"機器遺忘(Machine Unlearning)實驗報告\n")
            f.write(f"時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 2. 遺忘設定
            f.write("-"*80 + "\n")
            f.write("遺忘設定\n")
            f.write("-"*80 + "\n")
            if self.forget_classes is not None:
                f.write(f"遺忘類別: {sorted(self.forget_classes)}\n")
                f.write(f"保留類別數: {self.num_retain_classes}\n")
                f.write(f"總類別數: {self.total_classes}\n")
            else:
                f.write(f"遺忘樣本數: {len(self.forget_train_set)}\n")
                f.write(f"保留樣本數: {len(self.retain_train_set)}\n")
            f.write("\n")
            
            # 3. 模型架構
            f.write("-"*80 + "\n")
            f.write("模型架構參數\n")
            f.write("-"*80 + "\n")
            for k, v in self.model_args.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            # 4. 訓練設定
            f.write("-"*80 + "\n")
            f.write("訓練設定\n")
            f.write("-"*80 + "\n")
            f.write(f"設備: {self.device}\n")
            f.write(f"遺忘時間: {self.results['unlearning_time']:.2f} 秒\n")
            if 'retraining_time' in self.results:
                f.write(f"黃金標準重訓練時間: {self.results['retraining_time']:.2f} 秒\n")
            f.write("\n")
            
            # 5. 性能指標
            f.write("-"*80 + "\n")
            f.write("模型性能對比\n")
            f.write("-"*80 + "\n")
            f.write("原始模型:\n")
            f.write(f"  保留集準確率: {self.results['original_metrics'].get('retain_acc', 0):.2f}%\n")
            f.write(f"  遺忘集準確率: {self.results['original_metrics'].get('forget_acc', 0):.2f}%\n\n")
            
            f.write("遺忘後模型:\n")
            f.write(f"  保留集準確率: {self.results['unlearned_metrics'].get('retain_acc', 0):.2f}%\n")
            forget_acc = self.results['unlearned_metrics'].get('forget_acc', None)
            if forget_acc is not None:
                f.write(f"  遺忘集準確率: {forget_acc:.2f}%\n\n")
            else:
                f.write("  遺忘集準確率: N/A (已移除類別)\n\n")
            
            if 'retrained_metrics' in self.results:
                f.write("黃金標準模型:\n")
                f.write(f"  保留集準確率: {self.results['retrained_metrics'].get('retain_acc', 0):.2f}%\n")
                gs_forget_acc = self.results['retrained_metrics'].get('forget_acc', None)
                if gs_forget_acc is not None:
                    f.write(f"  遺忘集準確率: {gs_forget_acc:.2f}%\n\n")
                else:
                    f.write("  遺忘集準確率: N/A (已移除類別)\n\n")
            
            # 6. 遺忘指標
            f.write("-"*80 + "\n")
            f.write("機器遺忘指標\n")
            f.write("-"*80 + "\n")
            f.write(f"遺忘效果: {self.results.get('forget_effect', 0):.4f}\n")
            f.write(f"保留效果: {self.results.get('retain_effect', 0):.4f}\n")
            f.write(f"MU綜合分數: {self.results.get('MU_score', 0):.4f}\n")
            
            if 'gs_comparison' in self.results:
                f.write(f"黃金標準近似度: {self.results['gs_comparison'].get('gs_similarity', 0):.4f}\n")
                f.write(f"與黃金標準差異: {self.results['gs_comparison'].get('unl_vs_gs_diff', 0):.2f}%\n")
            f.write("\n")
            
            # 7. MIA結果
            if 'mia_results' in self.results and self.results['mia_results']:
                f.write("-"*80 + "\n")
                f.write("成員推斷攻擊(MIA)結果\n")
                f.write("-"*80 + "\n")
                f.write(f"總體攻擊準確率: {self.results['mia_results'].get('accuracy', 0):.4f}\n")
                f.write(f"保留集識別率: {self.results['mia_results'].get('retain_acc', 0):.4f}\n")
                f.write(f"遺忘集識別率: {self.results['mia_results'].get('forget_acc', 0):.4f}\n")
                f.write(f"MIA遺忘效果評分: {self.results['mia_results'].get('forget_effect', 0):.4f}\n\n")
                
                # 添加MIA分析結果
                f.write("MIA分析:\n")
                if self.results['mia_results'].get('forget_acc', 0) <= 0.55:
                    f.write("  遺忘集識別率接近隨機猜測水平，表明遺忘非常成功。\n")
                else:
                    f.write("  遺忘集識別率高於隨機猜測，表明遺忘可能不完全。\n")
                
                if self.results['mia_results'].get('retain_acc', 0) >= 0.9:
                    f.write("  保留集識別率很高，表明模型對保留數據的記憶很強，但這可能導致隱私風險。\n")
                f.write("\n")
            
            # 8. 結論與建議
            f.write("-"*80 + "\n")
            f.write("結論與建議\n")
            f.write("-"*80 + "\n")
            
            # 根據結果生成結論
            forget_effect = self.results.get('forget_effect', 0)
            retain_effect = self.results.get('retain_effect', 0)
            mu_score = self.results.get('MU_score', 0)
            
            if forget_effect > 0.95 and retain_effect >= 0:
                f.write("實驗結果表明，所採用的遺忘方法非常有效，完全移除了目標知識，同時保持了模型在保留數據上的性能。\n")
            elif forget_effect > 0.8:
                f.write("實驗結果表明，所採用的遺忘方法效果良好，大部分移除了目標知識。\n")
            else:
                f.write("實驗結果表明，所採用的遺忘方法效果有限，未能完全移除目標知識。建議嘗試其他遺忘策略。\n")
            
            if 'gs_comparison' in self.results:
                gs_similarity = self.results['gs_comparison'].get('gs_similarity', 0)
                if gs_similarity > 1.2:
                    f.write("值得注意的是，遺忘模型性能明顯優於黃金標準模型，這可能表明黃金標準模型訓練不足。建議增加黃金標準模型的訓練輪數或調整學習率。\n")
            
            if 'mia_results' in self.results:
                mia_forget_effect = self.results['mia_results'].get('forget_effect', 0)
                if mia_forget_effect < 0.7:
                    f.write("雖然從任務性能來看遺忘效果良好，但MIA評估表明隱私保護程度有限。建議考慮結合差分隱私技術以增強隱私保護。\n")
        
        print(f"詳細報告已保存至: {filepath}")
        return filepath

    def visualize_results(self):
        # 產生一張基本的柱狀圖：原始/遺忘/GS 的保留集 Acc
        try:
            names, accs = [], []
            if self.results.get('original_metrics'):
                names.append('Original'); accs.append(self.results['original_metrics'].get('retain_acc', 0))
            if self.results.get('unlearned_metrics'):
                names.append('Unlearned'); accs.append(self.results['unlearned_metrics'].get('retain_acc', 0))
            if self.results.get('retrained_metrics'):
                names.append('Gold'); accs.append(self.results['retrained_metrics'].get('retain_acc', 0))
            if names:
                plt.figure(figsize=(5,4))
                plt.bar(names, accs)
                plt.ylabel('Retain Test Acc (%)')
                plt.title('Retain Accuracy Comparison')
                out = os.path.join(self.output_dir, 'retain_acc_comparison.png')
                plt.tight_layout(); plt.savefig(out); plt.close()
                print(f"已輸出圖檔：{out}")
        except Exception as e:
            print(f"visualize_results 發生例外：{e}")


class MembershipInferenceAttack:
    """成員推斷攻擊(MIA)實現 - 評估模型是否真正"遺忘"了目標數據"""
    
    def __init__(self, target_model, device='cuda'):
        """
        初始化MIA評估器
        
        Args:
            target_model: 目標模型 (遺忘後的模型)
            device: 計算設備
        """
        self.target_model = target_model
        self.device = device
        self.attack_model = None
    
    def extract_features(self, model, loader, temperature: float = 3.0):
        """
        從目標模型中提取 MIA 特徵:
        - MSP (最大 softmax 機率, 經過溫度縮放)
        - Entropy (機率分布熵, 經過溫度縮放)
        - Margin (top1 - top2 機率差)
        - Per-sample CE Loss
        """
        model.eval()
        all_feats, all_labels = [], []
        ce = torch.nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                # 原始 logits
                logits = model(x)
                
                # Temperature scaling
                logits_T = logits / temperature
                probs_T = torch.softmax(logits_T, dim=1)

                # 基本特徵
                msp = probs_T.max(dim=1).values  # 最大機率
                entropy = -(probs_T * probs_T.clamp_min(1e-12).log()).sum(dim=1)
                loss = ce(logits, y)             # 注意: 用原始 logits 算 CE
                top2 = torch.topk(probs_T, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]

                # 堆成 feature vector: [msp, entropy, margin, loss]
                feats = torch.stack([msp, entropy, margin, loss], dim=1)

                all_feats.append(feats.cpu())
                all_labels.append(y.cpu())

        X = torch.cat(all_feats, dim=0).numpy()
        y = torch.cat(all_labels, dim=0).numpy()

        return X, y
    
    def prepare_attack_data(self, retain_loader, forget_loader, test_loader):
        """
        準備成員推斷攻擊所需的數據
        
        Args:
            retain_loader: 保留數據加載器
            forget_loader: 遺忘數據加載器
            test_loader: 測試數據加載器
        
        Returns:
            X_train, y_train, X_test, y_test, original_labels_test: 訓練和測試數據集
        """
        print("提取保留集特徵...")
        retain_X, retain_y = self.extract_features(self.target_model, retain_loader)

        print("提取遺忘集特徵...")
        forget_X, forget_y = self.extract_features(self.target_model, forget_loader)

        print("提取測試集特徵...")
        test_X, test_y = self.extract_features(self.target_model, test_loader)

        # 2) 成員標籤（攻擊目標）：retain=1, forget=0, test=0
        retain_m = np.ones(len(retain_X),  dtype=np.int32)          # 成員
        forget_m = np.zeros(len(forget_X), dtype=np.int32)          # 非成員（理應被忘）
        test_m   = np.zeros(len(test_X),   dtype=np.int32)          # 非成員

        # 📋 成員身份標籤分布
        total_members = len(retain_X)
        total_non_members = len(forget_X) + len(test_X)
        total_samples = total_members + total_non_members

        print(f"\n🏷️  成員身份標籤分布:")
        print(f"成員 (標籤=1): {total_members:,} 個 ({total_members/total_samples*100:.1f}%)")
        print(f"非成員 (標籤=0): {total_non_members:,} 個 ({total_non_members/total_samples*100:.1f}%)")
        print(f"成員:非成員比例 = 1:{total_non_members/total_members:.2f}")
        
        # 📈 各組詳細分解
        print(f"\n📈 各組詳細分解:")
        print(f"  保留集 (成員):     {len(retain_X):,} 個 → 標籤=1")
        print(f"  遺忘集 (非成員):   {len(forget_X):,} 個 → 標籤=0")
        print(f"  測試集 (非成員):   {len(test_X):,} 個 → 標籤=0")


        # 2) 來源旗標（僅用於評估分群）：
        #    保留/遺忘保留原始類別(>=0)；測試集標為負值以便在 evaluate 時用 <0 篩選
        retain_flag = retain_y.astype(np.int32)                 # >= 0
        forget_flag = forget_y.astype(np.int32)                 # >= 0
        test_flag   = (-test_y.astype(np.int32)) - 1

        # 4) ── 訓練集：只用 retain(=1) + test(=0) ────────────────────────────
        n_retain = int(len(retain_X) * 0.8)
        n_test   = int(len(test_X) * 0.8)
        X_train = np.vstack([retain_X[:n_retain], test_X[:n_test]])
        y_train = np.concatenate([retain_m[:n_retain], test_m[:n_test]]).astype(np.int32)
        
        # 5) ── 測試集：包含 retain + forget + test（按此順序串接） ───────────
        X_test = np.vstack([retain_X[n_retain:], forget_X, test_X[n_test:]])
        y_test = np.concatenate([retain_m[n_retain:], forget_m, test_m[n_test:]])
        original_labels_test = np.concatenate([retain_flag[n_retain:], forget_flag, test_flag[n_test:]])

        # 6) 訊息列印（方便檢查資料比例與語意）
        total_members = int(retain_m.sum())
        total_nonmembers = int((forget_m.sum() + test_m.sum()))
        total = total_members + total_nonmembers
        print("\n🏷️  成員身份標籤分布（整體語意）")
        print(f"  成員(=1):   {total_members:,} / {total:,} ({total_members/total*100:.1f}%)")
        print(f"  非成員(=0): {total_nonmembers:,} / {total:,} ({total_nonmembers/total*100:.1f}%)")
        print(f"  訓練集：retain(1)={len(retain_X):,}, test(0)={len(test_X):,}  → X_train={len(X_train):,}")
        print(f"  測試集：retain={len(retain_X):,}, forget={len(forget_X):,}, test={len(test_X):,} → X_test={len(X_test):,}")
        
        self.original_labels_test = original_labels_test

        return X_train, y_train, X_test, y_test, original_labels_test

    def train_attack_model(self, X_train, y_train, X_val, y_val,
                        epochs=50, learning_rate=1e-3, use_scheduler=True):
        """
        訓練 MIA 攻擊模型（驗證以 AUC 為主；不使用固定0.5）
        X_val/y_val 建議是 retain(1)+test(0) 的驗證切分；forget 僅作最終 evaluate_attack
        """
        # ---- Dataset / Loader ----
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

        # ---- 攻擊器（最後輸出 logits；不要 Sigmoid）----
        class AttackModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)  # logits
                )
            def forward(self, x):
                return self.net(x).view(-1)

        input_dim = X_train.shape[1]
        self.attack_model = AttackModel(input_dim).to(self.device)

        optimizer = optim.Adam(self.attack_model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # ---- 不平衡友善的 Loss ----
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5, verbose=True
            )
        else:
            scheduler = None

        best_auc = -1.0
        best_state = None
        best_thr = 0.5  # 只作紀錄用

        for epoch in range(epochs):
            # ---- Train ----
            self.attack_model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.attack_model(xb)               # logits
                loss = criterion(logits, yb)                 # BCEWithLogitsLoss
                loss.backward()
                optimizer.step()
                running += loss.item() * len(xb)
            train_loss = running / len(train_loader.dataset)

            # ---- Validate: 以 AUC 為主；同時計算最佳閾值的 Acc（只為參考，不回傳0.5）----
            self.attack_model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    lg = self.attack_model(xb)               # logits
                    all_logits.append(lg.cpu())
                    all_labels.append(yb)
            all_logits = torch.cat(all_logits).numpy()
            all_labels = torch.cat(all_labels).numpy()
            probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

            try:
                val_auc = roc_auc_score(all_labels, probs)
            except ValueError:
                # y_val 若不含兩類，AUC 無法計算；退回 0.5
                val_auc = 0.5

            # 找最佳閾值（Youden's J）僅作 log 參考
            try:
                fpr, tpr, th = roc_curve(all_labels, probs)
                j = (tpr - fpr)
                j_idx = int(j.argmax())
                thr = float(th[j_idx])
                y_bin = (probs >= thr).astype(int)
                val_acc_best = accuracy_score(all_labels, y_bin)
            except Exception:
                thr, val_acc_best = 0.5, 0.0

            if scheduler is not None:
                scheduler.step(val_auc)

            print(f"[Epoch {epoch+1:03d}] TrainLoss={train_loss:.4f}  ValAUC={val_auc:.4f}  "
                f"(best_thr~{thr:.3f}, Acc@best={val_acc_best:.4f})  LR={optimizer.param_groups[0]['lr']:.6f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_thr = thr
                best_state = copy.deepcopy(self.attack_model.state_dict())

        if best_state is not None:
            self.attack_model.load_state_dict(best_state)
            self.best_attack_threshold = best_thr  # 可供 evaluate_attack 使用或列印
            print(f"已恢復最佳攻擊器 (Val AUC={best_auc:.4f}, best_thr={best_thr:.3f})")

        return self.attack_model
    
    def evaluate_attack(self, X_test, y_test, original_labels_test):
        """
        在 retain / forget / test 上評估攻擊器效果
        Args:
            X_test: 測試集特徵 (numpy array, [N, d])
            y_test: membership 標籤 (1=retain, 0=forget/test)
            original_labels_test: 來源旗標 (>=0: retain/forget, <0: test)
        Returns:
            dict 包含總體與分組指標
        """
        self.attack_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            logits = self.attack_model(X_tensor)          # shape [N] 或 [N,1]
            if logits.ndim > 1:
                logits = logits.view(-1)                  # 強制 [N]
            probs = torch.sigmoid(logits).cpu().numpy()   # 轉成機率 [N]

        # ---- 保證 y_test 與 probs 對齊 ----
        y_test = np.array(y_test).ravel()
        probs = np.array(probs).ravel()
        assert probs.shape[0] == y_test.shape[0], \
            f"Shape mismatch: probs={probs.shape}, y_test={y_test.shape}"

        # --- 總體 AUC ---
        auc = roc_auc_score(y_test, probs)

        # --- 找最佳閾值 (Youden’s J index) ---
        fpr, tpr, thr = roc_curve(y_test, probs)
        youden_idx = np.argmax(tpr - fpr)
        best_thr = thr[youden_idx]

        # --- 總體 Accuracy @最佳閾值 ---
        y_pred = (probs >= best_thr).astype(int)
        acc = accuracy_score(y_test, y_pred)

        results = {
            "AUC": float(auc),
            "best_threshold": float(best_thr),
            "overall_acc": float(acc),
        }

        # --- 分組指標 ---
        group_metrics = {}
        groups = {
            "retain": (y_test == 1),
            "forget": ((y_test == 0) & (original_labels_test >= 0)),
            "test":   ((y_test == 0) & (original_labels_test < 0)),
        }

        for name, idx in groups.items():
            if idx.any():
                grp_acc = accuracy_score(y_test[idx], y_pred[idx])
                group_metrics[name] = {
                    "size": int(idx.sum()),
                    "acc": float(grp_acc)
                }

        results["groups"] = group_metrics

        print("\n📊 MIA 評估結果")
        print(f"  AUC = {auc:.4f}, best_thr = {best_thr:.4f}, Overall Acc = {acc:.4f}")
        for g, m in group_metrics.items():
            print(f"  {g:>6}: size={m['size']}, acc={m['acc']:.4f}")

        return results

    def plot_feature_distribution(self, features, labels, original_labels_test, title="MIA Feature Distribution", output_dir=None):
        """
        畫出 retain / forget / test 在不同特徵上的分布
        features: [N, d]，目前 d=4 (MSP, Entropy, Margin, Loss)
        labels: membership 標籤 (1=retain, 0=forget/test)
        """
        self.original_labels_test = original_labels_test

        retain_idx = (labels == 1)
        forget_idx = (labels == 0) & (self.original_labels_test >= 0)
        test_idx   = (labels == 0) & (self.original_labels_test < 0)

        names = ["MSP (confidence)", "Entropy", "Margin", "Loss"]
        num_feats = features.shape[1]

        plt.figure(figsize=(14, 10))
        for i in range(num_feats):
            plt.subplot((num_feats+1)//2, 2, i+1)
            if retain_idx.any():
                plt.hist(features[retain_idx, i], bins=40, alpha=0.5, label="Retain")
            if forget_idx.any():
                plt.hist(features[forget_idx, i], bins=40, alpha=0.5, label="Forget")
            if test_idx.any():
                plt.hist(features[test_idx, i], bins=40, alpha=0.5, label="Test")
            plt.title(names[i] if i < len(names) else f"Feature {i}")
            plt.legend()

        plt.suptitle(title)
        plt.tight_layout()

        if output_dir:
            save_path = os.path.join(output_dir, f"{title.replace(' ','_')}.png")
            plt.savefig(save_path)
            print(f"Feature distribution saved to {save_path}")
        else:
            plt.show()

def main():
    # 命令行參數設置
    parser = argparse.ArgumentParser(description='機器遺忘(Machine Unlearning)實驗')
    parser.add_argument("--num_epochs", type=int, default=80, help="遺忘訓練的輪數")
    parser.add_argument("--gs_epochs", type=int, default=250, help="黃金標準模型訓練的輪數")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="遺忘訓練的初始學習率")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="遺忘訓練的最小學習率")
    parser.add_argument("--lr_scheduler", type=str, choices=['cosine', 'step', 'plateau', 'onecycle'], 
                       default='cosine', help="學習率調度器類型")
    parser.add_argument("--gs_learning_rate", type=float, default=1e-3, help="黃金標準模型的初始學習率")
    parser.add_argument("--gs_lr_min", type=float, default=1e-6, help="黃金標準模型的最小學習率")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=['cosine', 'onecycle', 'step'], 
                       default='onecycle', help="黃金標準模型的學習率調度器類型")
    parser.add_argument("--mia_learning_rate", type=float, default=1e-3, help="MIA模型的學習率")
    parser.add_argument("--mia_epochs", type=int, default=50, help="MIA訓練輪數")
    parser.add_argument("--mia_use_scheduler", type=bool, default=True, help="是否對MIA模型使用學習率調度器")
    parser.add_argument("--output_dir", type=str, default="./unlearning_output", help="輸出目錄")
    parser.add_argument("--best_model_path", type=str, 
                      default="/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth", 
                      help="預訓練模型路徑")
    parser.add_argument("--unlearning_method", type=str, 
                      choices=['head_retrain', 'negative_gradient'], 
                      default='head_retrain', 
                      help="遺忘方法選擇")
    parser.add_argument("--forget_classes", type=str, default="20-29", 
                      help="要遺忘的類別 (e.g., '20-29' or '20,21,22')")
    parser.add_argument("--no_mia", action="store_false", dest="run_mia", 
                    default=True, help="設置此參數禁用MIA評估")
    args = parser.parse_args()
    
    # 處理遺忘類別參數
    if '-' in args.forget_classes:
        start, end = map(int, args.forget_classes.split('-'))
        forget_classes = set(range(start, end+1))
    else:
        forget_classes = set(map(int, args.forget_classes.split(',')))
    
    # 模型參數
    model_args = {
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
    
    # 創建實驗輸出目錄
    timestamp = time.strftime("%Y%m%d-%H%M")
    output_dir = f"{args.output_dir}/{args.unlearning_method}_forget{args.forget_classes}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存命令行參數
    with open(f"{output_dir}/args.txt", 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # 初始化機器遺忘框架
    mul = MachineUnlearning(
        model_class=VisionTransformer,
        model_args=model_args,
        device='cuda',
        output_dir=output_dir
    )
    
    # 載入預訓練模型
    mul.load_original_model(args.best_model_path)
    
    # 準備數據
    mul.prepare_data(dataset_name='CIFAR100', forget_classes=forget_classes)
    
    # 評估原始模型性能
    print("\n1. 評估原始模型")
    mul.evaluate_all_models()
    
    # 選擇遺忘方法
    print(f"\n2. 執行機器遺忘 - {args.unlearning_method}")
    if args.unlearning_method == 'head_retrain':
        mul.unlearn_by_retraining_head(
            epochs=args.num_epochs, 
            lr=args.learning_rate, 
            lr_scheduler_type=args.lr_scheduler, 
            min_lr=args.lr_min
        )
    elif args.unlearning_method == 'negative_gradient':
        mul.unlearn_by_negative_gradient(
            epochs=args.num_epochs, 
            lr=args.learning_rate,
            retain_epochs=args.num_epochs//3,
            lr_scheduler_type=args.lr_scheduler,
            min_lr=args.lr_min
        )
    
    # 訓練黃金標準
    print("\n3. 訓練黃金標準模型")
    mul.train_gold_standard(
        epochs=args.gs_epochs, 
        lr=args.gs_learning_rate,
        lr_scheduler_type=args.gs_lr_scheduler,
        min_lr=args.gs_lr_min
    )
    
    # 評估所有模型性能
    print("\n4. 評估所有模型性能")
    mul.evaluate_all_models()
    
    # 計算遺忘指標
    print("\n5. 計算機器遺忘指標")
    mul.calculate_metrics()
    
    # 運行MIA攻擊評估
    if args.run_mia:
        print("\n6. 執行成員推斷攻擊(MIA)評估")
        mul.run_mia(
            train_epochs=args.mia_epochs,
            learning_rate=args.mia_learning_rate,
            use_scheduler=args.mia_use_scheduler
        )
    
    # 可視化結果
    print("\n7. 可視化結果")
    mul.visualize_results()
    
    # 保存詳細報告
    print("\n8. 保存詳細報告")
    mul.save_complete_report()
    
    # 保存結果
    print("\n9. 保存結果")
    mul.save_results()
    
    print("\n機器遺忘實驗完成!")
    print(f"所有結果保存在：{output_dir}")


if __name__ == "__main__":
    main()