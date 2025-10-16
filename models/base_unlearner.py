"""
基礎遺忘器類 - 提供所有遺忘方法的共同接口
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseUnlearner(ABC):
    """
    遺忘模型基類 - 實現各種遺忘方法的共同接口
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    @abstractmethod
    def unlearn(self, *args, **kwargs):
        """
        抽象方法：實現具體的遺忘策略
        """
        pass
    
    def _evaluate_retain(self, model, retain_loader, class_mapping=None):
        """評估模型在保留集上的性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 如果有類別映射，將標籤映射到新類別空間
                if class_mapping is not None:
                    mapped_labels = []
                    for label in labels:
                        if label.item() in class_mapping:
                            mapped_labels.append(class_mapping[label.item()])
                        else:
                            continue  # 跳過不在映射中的標籤
                    
                    if not mapped_labels:
                        continue
                    
                    labels = torch.tensor(mapped_labels, device=self.device)
                    inputs = inputs[:len(mapped_labels)]
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0
    
    def get_scheduler(self, optimizer, lr_scheduler_type, epochs, dataloader, min_lr):
        """獲取學習率調度器"""
        if lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        elif lr_scheduler_type == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=optimizer.param_groups[0]['lr'], 
                total_steps=epochs * len(dataloader),
                pct_start=0.2, anneal_strategy='cos'
            )
        elif lr_scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=min_lr
            )
        elif lr_scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        else:
            raise ValueError(f"不支持的調度器類型: {lr_scheduler_type}")
