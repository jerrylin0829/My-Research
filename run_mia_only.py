"""
只執行 MIA（成員推斷攻擊）分析，不跑遺忘訓練。
請確認已經有訓練好的模型權重（如 best_vit_20250423-0016.pth）和 CIFAR-100 數據集。
"""
import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from mul_vit_1 import VisionTransformer, MembershipInferenceAttack

# ====== 參數設定 ======
model_path = "/home/davidhuang/vits-for-small-scale-datasets/checkpoints/ViT_classattn_CIFAR100/BEST_ViT_20250423-0016_lr0.001_bs256_epochs600/best_vit_20250423-0016.pth"
output_dir = "./mia_only_output"
batch_size = 128
num_workers = 4
img_size = 32
num_classes = 100

os.makedirs(output_dir, exist_ok=True)

# ====== 模型架構參數 ======
model_args = {
    'img_size': img_size,
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': num_classes,
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

# ====== 載入模型 ======
model = VisionTransformer(**model_args)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to('cuda')
model.eval()

# ====== 準備資料集 ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

retain_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
forget_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 這裡 retain/forget 可根據你的需求分割（例如前 10 類為 retain，後 10 類為 forget）
forget_classes = set(range(10, 20))
retain_classes = set(range(100)) - forget_classes

retain_indices = [i for i, (_, label) in enumerate(retain_dataset) if label in retain_classes]
forget_indices = [i for i, (_, label) in enumerate(forget_dataset) if label in forget_classes]

retain_subset = torch.utils.data.Subset(retain_dataset, retain_indices)
forget_subset = torch.utils.data.Subset(forget_dataset, forget_indices)

retain_loader = DataLoader(retain_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
forget_loader = DataLoader(forget_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ====== 執行 MIA ======
mia = MembershipInferenceAttack(model, device='cuda')
X_train, y_train, X_test, y_test, original_labels_test = mia.prepare_attack_data(
    retain_loader, forget_loader, test_loader
)

# ====== 保存特徵分布圖 ======
mia.plot_feature_distribution(
    np.vstack([X_train, X_test]),
    np.concatenate([y_train, y_test]),
    title="MIA Feature Distribution",
    output_dir=output_dir
)

# ====== 訓練攻擊模型 ======
mia.train_attack_model(
    X_train, y_train, X_test, y_test,
    epochs=50, learning_rate=1e-3, use_scheduler=True
)

# ====== 評估攻擊效果 ======
results = mia.evaluate_attack(X_test, y_test, original_labels_test)
print("\n===== MIA Results =====")
for k, v in results.items():
    print(f"{k}: {v}")

print(f"分布圖已保存於: {output_dir}")
