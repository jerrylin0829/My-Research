import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
import os, json
from utils.attention_viz import plot_head_relevance_heatmap, dump_attention_maps, plot_attention_maps_compare

class AHRCEUnlearner:
    def __init__(self, model, device, retain_loader, forget_loader, cfg):
        self.model = model
        self.device = device
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.cfg = cfg
        self._prepare_head_scale_params()

    # 1) 在 attn 注入可學縮放 γ_{l,h}
    def _prepare_head_scale_params(self):
        self.scalers = []  # 收集需要學習的縮放張量
        for b, block in enumerate(self.model.blocks):
            if hasattr(block, 'attn'):
                H = block.attn.num_heads
                scale = nn.Parameter(torch.ones(H, device=self.device))
                block.attn.register_parameter("head_scale", scale)
                self.scalers.append(scale)

    @torch.no_grad()
    def _compute_head_relevance(self):
        # 跟你現有版本邏輯一致：統計 retain/forget 的 head 指標 → ratio
        # 這裡示意：用 CLS row 的平均注意力作為 head score
        def collect(loader):
            sums = defaultdict(float); cnts = defaultdict(int)
            for x, _ in loader:
                x = x.to(self.device)
                attn_list = self._forward_collect_attn(x)  # list[(layer, attn[B,H,S,S])]
                for l, A in attn_list:
                    # 取 CLS 的 row 平均（可換成 top-k mass）
                    val = A[:, :, 0, :].mean(dim=(0,2))  # [H]
                    for h, v in enumerate(val.tolist()):
                        sums[(l,h)] += v; cnts[(l,h)] += 1
            return {k: sums[k]/max(1,cnts[k]) for k in sums}

        retain = collect(self.retain_loader)
        forget = collect(self.forget_loader)

        relevance = {}
        for k in retain:
            r = retain.get(k, 1e-6); f = forget.get(k, 1e-6)
            relevance[k] = (f + 1e-6) / (r + 1e-6)
        return relevance

    def _forward_collect_attn(self, x):
        attn_cache = []
        hooks = []
        for l, block in enumerate(self.model.blocks):
            if hasattr(block, 'attn'):
                def make_hook(li):
                    def _hook(m, i, o):
                        if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                            attn_cache.append((li, o[1].detach()))  # [B,H,S,S]
                    return _hook
                hooks.append(block.attn.register_forward_hook(make_hook(l)))
        with torch.no_grad():
            _ = self.model(x)
        for h in hooks: h.remove()
        return attn_cache

    def unlearn(self, out_dir):
        self.model.to(self.device)
        self.model.train()

        # 2) 計算 relevance，選頭
        rel = self._compute_head_relevance()
        heads_sorted = sorted(rel.items(), key=lambda kv: kv[1], reverse=True)
        keep = int(self.cfg.get("top_percent", 0.3) * len(heads_sorted))
        target_heads = set([k for k,_ in heads_sorted[:keep]])

        # 視覺化 relevance
        viz_dir = os.path.join(out_dir, "viz_attention")
        os.makedirs(viz_dir, exist_ok=True)
        plot_head_relevance_heatmap(rel, len(self.model.blocks), self.model.blocks[0].attn.num_heads,
                                    os.path.join(viz_dir, "head_relevance_heatmap.png"))

        # 3) 訓練（AHRCE 搭配 retain/forget 雙路）
        ref = self.cfg.get("ref_model")  # 已凍結的原始模型
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      lr=self.cfg.get("lr", 3e-5), weight_decay=1e-4)

        # 僅允許 attn 層與 head_scale 更新（可選）
        if self.cfg.get("only_attn", True):
            for name, p in self.model.named_parameters():
                p.requires_grad = ("attn" in name) or ("head" in name)  # head=分類頭
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.get("ls", 0.1))
        lambda_c = self.cfg.get("lambda_c", 0.5)
        lambda_e = self.cfg.get("lambda_e", 0.2)
        lambda_u = self.cfg.get("lambda_u", 0.2)
        lambda_g = self.cfg.get("lambda_gamma", 5e-4)
        T = self.cfg.get("temp", 3.0)

        retain_iter = iter(self.retain_loader)
        forget_iter = iter(self.forget_loader)
        steps = self.cfg.get("steps", 1000)

        for step in range(steps):
            try:
                xr, yr = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.retain_loader); xr, yr = next(retain_iter)
            try:
                xf, _ = next(forget_iter)
            except StopIteration:
                forget_iter = iter(self.forget_loader); xf, _ = next(forget_iter)

            xr, yr = xr.to(self.device), yr.to(self.device)
            xf = xf.to(self.device)

            # ===== retain path =====
            logits_r, feats_r, attn_r = self._forward_with_attn(xr)
            L_cls = criterion(logits_r, yr)
            
            # 🛠️ 修復：處理 ref_model 為 None 的情況
            if ref is not None:
                with torch.no_grad():
                    _, feats_ref, attn_ref = self._forward_with_attn(xr, model_ref=ref)
                L_consistency = F.mse_loss(feats_r, feats_ref)
            else:
                # 如果沒有參考模型，設置為0
                L_consistency = torch.tensor(0.0, device=self.device)

            # ===== forget path =====
            logits_f, _, attn_f = self._forward_with_attn(xf)
            
            # attn entropy on target heads
            ent = torch.tensor(0.0, device=self.device)
            count = 0
            for (l, A) in attn_f:  # A: [B,H,S,S]
                H = -(A * (A.clamp_min(1e-12)).log()).sum(dim=-1).mean(dim=(0,2))  # [H]
                for h in range(H.shape[0]):
                    if (l,h) in target_heads:
                        ent = ent + (-H[h])  # maximize entropy -> minimize -H
                        count += 1

            if count > 0:
                L_attn_entropy = ent / count
            else:
                L_attn_entropy = torch.tensor(0.0, device=self.device)

            # uniform logits on forget
            p = torch.softmax(logits_f / T, dim=1)
            u = torch.full_like(p, 1.0 / p.size(1))
            L_uniform = F.kl_div(p.log(), u, reduction='batchmean')

            # head-scale sparsity
            L_gamma = torch.tensor(0.0, device=self.device)
            gamma_count = 0

            for b, block in enumerate(self.model.blocks):
                if hasattr(block.attn, "head_scale"):
                    g = block.attn.head_scale
                    # 只對 target_heads 施以 L1
                    mask = torch.ones_like(g)
                    for h in range(g.numel()):
                        if (b, h) not in target_heads:
                            mask[h] = 0.0
                    L_gamma = L_gamma + (mask * g.abs()).sum() / (mask.sum().clamp_min(1.0))

            if gamma_count > 0:
                L_gamma = L_gamma / gamma_count

            loss = L_cls + lambda_c*L_consistency + lambda_e*L_attn_entropy + lambda_u*L_uniform + lambda_g*L_gamma
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Optional：把 head_scale 限制在 [0,1]
            for b, block in enumerate(self.model.blocks):
                if hasattr(block.attn, "head_scale"):
                    block.attn.head_scale.data.clamp_(0.0, 1.0)

            if step % 50 == 0:
                def safe_item(tensor_or_float):
                    """安全地獲取數值，無論是 tensor 還是 float"""
                    if isinstance(tensor_or_float, torch.Tensor):
                        return tensor_or_float.item()
                    else:
                        return float(tensor_or_float)

                print(f"[AHRCE] step {step}: "
                      f"L_cls={safe_item(L_cls):.3f} L_c={safe_item(L_consistency):.3f} "
                      f"L_e={safe_item(L_attn_entropy):.3f} L_u={safe_item(L_uniform):.3f} L_g={safe_item(L_gamma):.3f}")
    
        # 前/後注意力對比（小批）
        try:
            before_maps = dump_attention_maps(self.model, self.forget_loader, self.device, max_batches=1)
            after_maps  = dump_attention_maps(self.model, self.forget_loader, self.device, max_batches=1)
            plot_attention_maps_compare(before_maps, after_maps, os.path.join(viz_dir, "maps_compare"))
        except Exception as e:
            print(f"⚠️ 注意力可視化失敗: {e}")

        # 輸出 head_scale
        gamma_json = {}
        for b, block in enumerate(self.model.blocks):
            if hasattr(block.attn, "head_scale"):
                gamma_json[f"layer{b}"] = block.attn.head_scale.detach().cpu().tolist()
        
        with open(os.path.join(viz_dir, "head_scale.json"), "w") as f:
            json.dump(gamma_json, f, indent=2)

        return self.model

    def _forward_with_attn(self, x, model_ref=None):
        m = self.model if model_ref is None else model_ref
        attn_cache = []
        feats_last = None

        hooks = []
        for l, block in enumerate(m.blocks):
            if hasattr(block, 'attn'):
                def make_hook(li):
                    def _hook(module, inp, out):
                        if isinstance(out, tuple) and len(out) >= 2:
                            attn_cache.append((li, out[1]))
                    return _hook
                hooks.append(block.attn.register_forward_hook(make_hook(l)))
        out = m(x)  # 假設 m 回傳 logits
        for h in hooks: h.remove()

        # 如果你有方便的中間特徵接口：feats_last = m.get_last_mlp(x)
        # 這裡簡化用 logits 當作 feature（或自行改為倒數第二層的投影）
        feats_last = out

        return out, feats_last, attn_cache
