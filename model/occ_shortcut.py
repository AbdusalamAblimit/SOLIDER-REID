import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """梯度反转层：前向恒等，反向把梯度乘以 -alpha。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.alpha), None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class OccluderHead(nn.Module):
    """预测无遮挡物或遮挡物池中的具体 patch 编号。"""

    def __init__(self, in_dim, pool_size, hidden_dim=None):
        super(OccluderHead, self).__init__()
        if pool_size < 1:
            raise ValueError("OSS.POOL_SIZE must be >= 1")
        hidden_dim = hidden_dim or max(128, min(512, in_dim // 2))
        self.pool_size = int(pool_size)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.pool_size + 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.classifier(x)


def _sample_patch_hw(height, width, min_scale=0.2, max_scale=0.5, rng=None):
    rng = rng or random
    scale_h = rng.uniform(min_scale, max_scale)
    scale_w = rng.uniform(min_scale, max_scale)
    patch_h = max(1, min(height, int(round(height * scale_h))))
    patch_w = max(1, min(width, int(round(width * scale_w))))
    return patch_h, patch_w


def build_occluder_pool(images, pool_size=256, pool=None, min_scale=0.2, max_scale=0.5, rng=None):
    """从给定图像张量裁 patch，随机源可由调用方显式传入。"""
    rng = rng or random
    if pool is None:
        pool = []
    if pool_size <= 0 or len(pool) >= pool_size:
        return pool
    if images.dim() != 4:
        raise ValueError("images must be a tensor with shape (B, C, H, W)")

    batch_size, _, height, width = images.shape
    if batch_size == 0:
        return pool

    while len(pool) < pool_size:
        img_idx = rng.randrange(batch_size)
        patch_h, patch_w = _sample_patch_hw(height, width, min_scale, max_scale, rng)
        top = rng.randint(0, height - patch_h)
        left = rng.randint(0, width - patch_w)
        patch = images[img_idx, :, top:top + patch_h, left:left + patch_w].detach().clone()
        pool.append(patch)
    return pool


def _extract_image_tensor(sample):
    if isinstance(sample, (tuple, list)):
        sample = sample[0]
    if not torch.is_tensor(sample) or sample.dim() != 3:
        raise ValueError("image_source must return a C,H,W tensor or a tuple whose first item is one")
    return sample


def build_fixed_occluder_pool(image_source, pool_size=256, seed=20260606, min_scale=0.2, max_scale=0.5):
    """启动 DataLoader worker 前，从训练图确定性裁出固定遮挡物池。"""
    if pool_size <= 0:
        return []
    if len(image_source) == 0:
        raise ValueError("cannot build OSS occluder pool from an empty training set")

    rng = random.Random(int(seed))
    pool = []
    while len(pool) < pool_size:
        img_idx = rng.randrange(len(image_source))
        image = _extract_image_tensor(image_source[img_idx])
        build_occluder_pool(
            image.unsqueeze(0),
            pool_size=len(pool) + 1,
            pool=pool,
            min_scale=min_scale,
            max_scale=max_scale,
            rng=rng,
        )
    return pool


def randomize_occluder_labels(occ_id, pool_size, rng=None):
    """只替换已贴 patch 的非 0 标签，0 仍表示无遮挡。"""
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    rng = rng or random
    randomized = occ_id.clone()
    flat = randomized.view(-1)
    for i in range(flat.numel()):
        if int(flat[i].item()) != 0:
            flat[i] = rng.randint(1, int(pool_size))
    return randomized


def paste_occluder_batch(images, pool, aug_prob=0.3, rng=None, random_label=False, label_rng=None,
                         return_metadata=False):
    """把池中随机 patch 贴到训练图上，并返回 occ_id，0 表示未贴。

    return_metadata=True 时额外返回归一化矩形和证据分 e=1-遮挡面积占比。默认仍返回
    原来的二元组，保证 OSS 既有调用不变。
    """
    rng = rng or random
    if images.dim() != 4:
        raise ValueError("images must be a tensor with shape (B, C, H, W)")

    out = images.clone()
    occ_id = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
    rects = torch.zeros((images.shape[0], 4), dtype=torch.float32, device=images.device)
    evidence = torch.ones(images.shape[0], dtype=torch.float32, device=images.device)
    if aug_prob <= 0 or not pool:
        if return_metadata:
            return out, occ_id, rects, evidence
        return out, occ_id

    _, _, height, width = images.shape
    for i in range(images.shape[0]):
        if rng.random() >= aug_prob:
            continue
        patch_idx = rng.randrange(len(pool))
        patch = pool[patch_idx].to(device=out.device, dtype=out.dtype)
        patch_h, patch_w = patch.shape[-2:]
        if patch_h > height or patch_w > width:
            new_h, new_w = min(patch_h, height), min(patch_w, width)
            patch = F.interpolate(patch.unsqueeze(0), size=(new_h, new_w),
                                  mode='bilinear', align_corners=False).squeeze(0)
            patch_h, patch_w = patch.shape[-2:]
        top = rng.randint(0, height - patch_h)
        left = rng.randint(0, width - patch_w)
        out[i, :, top:top + patch_h, left:left + patch_w] = patch
        occ_id[i] = patch_idx + 1
        rects[i] = torch.tensor(
            [top / height, left / width, (top + patch_h) / height, (left + patch_w) / width],
            dtype=torch.float32,
            device=images.device,
        )
        evidence[i] = 1.0 - float(patch_h * patch_w) / float(height * width)
    if random_label:
        occ_id = randomize_occluder_labels(occ_id, len(pool), label_rng)
    if return_metadata:
        return out, occ_id, rects, evidence
    return out, occ_id


def occluder_shortcut_loss(head, global_feat, occ_id, alpha):
    if isinstance(global_feat, (list, tuple)):
        global_feat = global_feat[0]
    occ_logits = head(grad_reverse(global_feat, alpha))
    return occ_logits, F.cross_entropy(occ_logits, occ_id)
