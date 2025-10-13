"""
Attention visualization utilities
- plot_head_relevance_heatmap: heatmap of (layer x head) relevance scores
- dump_attention_maps: run model on a few batches and collect attention weights per layer
- plot_attention_maps_compare: compare before/after attention maps and save images

Compatibility:
- Expects ViT-like model with attribute `blocks` iterable.
- Each block should expose its attention module as `block.attn`.
- The attention module forward hook is expected to receive output where out[1] contains attn weights [B,H,S,S]
  If your module stores attn in a different attr (e.g., attn_weights), this code tries common fallbacks.
"""
from __future__ import annotations
import os
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_head_relevance_heatmap(relevance_scores: Dict[Tuple[int,int], float],
                                num_layers: int, num_heads: int, save_path: str):
    """
    Draw a heatmap of relevance scores (forget / retain) per (layer, head).

    Args:
        relevance_scores: dict[(layer_idx, head_idx)] -> float
        num_layers: number of transformer blocks
        num_heads: number of heads per block (assumes constant)
        save_path: path to save PNG
    """
    Hmat = np.zeros((num_layers, num_heads), dtype=float)
    for (l, h), v in relevance_scores.items():
        if 0 <= l < num_layers and 0 <= h < num_heads:
            Hmat[l, h] = float(v)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(max(6, num_heads*0.5), max(4, num_layers*0.4)))
    im = plt.imshow(Hmat, cmap="magma", aspect="auto")
    plt.colorbar(im, label="forget / retain relevance")
    plt.xlabel("Head index")
    plt.ylabel("Layer index")
    plt.title("Attention Head Relevance (forget vs retain)")
    plt.xticks(np.arange(num_heads))
    plt.yticks(np.arange(num_layers))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def dump_attention_maps(model: torch.nn.Module, dataloader, device: torch.device,
                        max_batches: int = 2) -> List[Tuple[int, np.ndarray]]:
    """
    Run the model on a few batches and collect attention weights.

    Args:
        model: ViT-like model with .blocks where each block has .attn
        dataloader: DataLoader yielding (x, y)
        device: torch device
        max_batches: how many batches to sample (small number)

    Returns:
        list of (layer_idx, attn_mean_np) where attn_mean_np shape is [H, S, S]
    """
    model = model.to(device)
    model.eval()

    attn_cache = defaultdict(list)
    hooks = []

    def make_hook(li):
        def _hook(module, inputs, output):
            # Try common patterns: output is (out, attn_weights) OR module has attn_weights attr
            attn = None
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn = output[1]
            elif hasattr(module, "attn_weights"):
                attn = getattr(module, "attn_weights")
            # Some implementations may store attention in output.attn_map etc - not handled here
            if attn is not None:
                # ensure detach and cpu
                if isinstance(attn, torch.Tensor):
                    attn_cache[li].append(attn.detach().cpu())
                else:
                    # if attn is list/tuple
                    try:
                        t = torch.stack([torch.as_tensor(a) for a in attn], dim=0)
                        attn_cache[li].append(t.cpu())
                    except Exception:
                        pass
        return _hook

    # register hooks on blocks that have .attn
    if not hasattr(model, "blocks"):
        raise RuntimeError("Model does not have attribute 'blocks' (expected ViT-like).")

    for li, block in enumerate(model.blocks):
        if hasattr(block, "attn"):
            try:
                hooks.append(block.attn.register_forward_hook(make_hook(li)))
            except Exception:
                # some attn modules may not accept hooks; skip quietly
                pass

    # Run a few batches
    it = iter(dataloader)
    for bi in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0].to(device)
        try:
            _ = model(x)
        except Exception:
            # some models return tuple or require extra args; try to call safely
            try:
                _ = model.forward(x)
            except Exception as e:
                # cleanup hooks and raise
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
                raise RuntimeError(f"Model forward failed inside dump_attention_maps: {e}")

    # remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # aggregate per layer
    out = []
    for li, tensors in attn_cache.items():
        if len(tensors) == 0:
            continue
        try:
            # tensors: list of [B, H, S, S] -> cat -> [B_total, H, S, S] -> mean over batch
            cat = torch.cat(tensors, dim=0)    # [B_total, H, S, S]
            mean_attn = cat.mean(dim=0)        # [H, S, S]
            out.append((li, mean_attn.numpy()))
        except Exception:
            # fallback: try stacking as numpy
            try:
                np_list = [np.asarray(t) for t in tensors]
                mean_attn = np.mean(np.concatenate(np_list, axis=0), axis=0)
                out.append((li, mean_attn))
            except Exception:
                pass
    return sorted(out, key=lambda x: x[0])


def _safe_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def plot_attention_maps_compare(before_maps: List[Tuple[int, np.ndarray]],
                                after_maps: List[Tuple[int, np.ndarray]],
                                save_dir: str,
                                heads_to_plot: int = 4):
    """
    Compare before/after attention maps and save images.
    For each layer present in both lists, we will plot up to `heads_to_plot` heads:
      [before head0] [after head0]  (side-by-side)
      [before head1] [after head1]
      ...

    Args:
        before_maps: list[(layer_idx, attn_np)] attn_np shape [H, S, S]
        after_maps: list[(layer_idx, attn_np)]
        save_dir: directory to write images (will be created)
        heads_to_plot: number of heads to show per layer (from head 0)
    """
    os.makedirs(save_dir, exist_ok=True)

    # convert to dict by layer
    bdict = {li: _safe_to_numpy(A) for li, A in before_maps}
    adict = {li: _safe_to_numpy(A) for li, A in after_maps}
    layers = sorted(set(bdict.keys()) & set(adict.keys()))

    if len(layers) == 0:
        # nothing to plot
        return

    for li in layers:
        A0 = bdict[li]  # [H,S,S]
        A1 = adict[li]
        # check dims
        if A0.ndim != 3 or A1.ndim != 3:
            continue
        H = min(A0.shape[0], A1.shape[0])
        heads = min(heads_to_plot, H)

        # For each head, produce a side-by-side figure
        for h in range(heads):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            vmin = min(A0[h].min(), A1[h].min())
            vmax = max(A0[h].max(), A1[h].max())
            im0 = axs[0].imshow(A0[h], vmin=vmin, vmax=vmax, cmap="viridis")
            axs[0].set_title(f"Layer {li} Head {h} BEFORE")
            plt.colorbar(im0, ax=axs[0])
            im1 = axs[1].imshow(A1[h], vmin=vmin, vmax=vmax, cmap="viridis")
            axs[1].set_title(f"Layer {li} Head {h} AFTER")
            plt.colorbar(im1, ax=axs[1])
            diff = np.abs(A1[h] - A0[h])
            im2 = axs[2].imshow(diff, cmap="magma")
            axs[2].set_title(f"Layer {li} Head {h} ABS DIFF")
            plt.colorbar(im2, ax=axs[2])
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"layer{li:02d}_head{h}_before_after_diff.png")
            plt.savefig(save_path, dpi=200)
            plt.close()

        # Additionally, save a grid overview of first `heads` heads (before / after)
        ncol = heads
        fig, axs = plt.subplots(2, ncol, figsize=(3*ncol, 6))
        for h in range(heads):
            axs[0, h].imshow(A0[h], cmap="viridis")
            axs[0, h].set_title(f"B L{li} H{h}")
            axs[0, h].axis("off")
            axs[1, h].imshow(A1[h], cmap="viridis")
            axs[1, h].set_title(f"A L{li} H{h}")
            axs[1, h].axis("off")
        plt.suptitle(f"Layer {li} - BEFORE (top) / AFTER (bottom)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        grid_path = os.path.join(save_dir, f"layer{li:02d}_overview.png")
        plt.savefig(grid_path, dpi=200)
        plt.close()
