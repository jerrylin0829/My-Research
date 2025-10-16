import os
import sys
import torch
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_LSA_classattn import VisionTransformer
from mul_vit_1 import MachineUnlearning

def run_training_trial(args):
    """
    執行單次的 GS 訓練，並返回最終準確率
    """

    print(f"===== 開始為 forget_count={args.forget_count} 訓練黃金標準模型 =====")
    
    random.seed(42 + args.forget_count)
    forget_classes = set(random.sample(range(100), args.forget_count))
    
    mul = MachineUnlearning(
        model_class=VisionTransformer,
        model_args={
            'img_size': 32, 'patch_size': 4, 'in_chans': 3, 'num_classes': 100,
            'embed_dim': 300, 'depth': 10, 'num_heads': 12, 'mlp_ratio': 4.0,
            'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0,
            'drop_path_rate': 0.05, 'is_LSA': True
        },
        device=args.device,
        output_dir=os.path.join(args.output_dir, f"temp_training_forget_{args.forget_count}"),
        args=args
    )
    
    mul.prepare_data(dataset_name='CIFAR100', forget_classes=forget_classes)
    
    mul.train_gold_standard(
        epochs=args.gs_epochs,
        lr=args.gs_lr,
        lr_scheduler_type=args.gs_scheduler,
        min_lr=args.gs_min_lr,
        weight_decay=args.gs_weight_decay,
        use_mixup=args.use_mixup,
        mix_alpha=args.mix_alpha,
        label_smoothing=args.label_smoothing,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    gs_model_path = os.path.join(args.output_dir, f"gs_model_forget{args.forget_count}.pth")
    torch.save(mul.retrained_model.state_dict(), gs_model_path)
    print(f"✅ 黃金標準模型已保存至: {gs_model_path}")

    final_acc = mul.results['retrained_metrics'].get('retain_acc', 0.0)
    return final_acc
