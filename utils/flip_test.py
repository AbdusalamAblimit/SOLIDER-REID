"""Flip-test helper for default eval (equal_concat + flip)."""
import torch
from datasets.pose_dataset import FLIP_PAIRS


def flip_batch(imgs, pose_dict):
    """Horizontal flip images + pose heatmaps + keypoints with L-R swap."""
    imgs_flip = imgs.flip(-1)

    if pose_dict is None:
        return imgs_flip, None

    pd = {}
    for k, v in pose_dict.items():
        pd[k] = v.clone() if isinstance(v, torch.Tensor) else v

    # heatmaps: (B, max_persons, 17, H, W) → flip W + swap L/R channels
    if 'heatmaps' in pd:
        hm = pd['heatmaps'].flip(-1)
        for l, r in FLIP_PAIRS:
            hm[:, :, [l, r]] = hm[:, :, [r, l]]
        pd['heatmaps'] = hm

    # keypoints: (B, max_persons, 17, 2) → x mirror + swap L/R
    if 'keypoints' in pd:
        kp = pd['keypoints'].clone()
        img_w = imgs.shape[-1]
        kp[..., 0] = img_w - 1 - kp[..., 0]
        for l, r in FLIP_PAIRS:
            kp[:, :, [l, r]] = kp[:, :, [r, l]]
        pd['keypoints'] = kp

    # scores: (B, max_persons, 17) → swap L/R
    if 'scores' in pd:
        sc = pd['scores'].clone()
        for l, r in FLIP_PAIRS:
            sc[:, :, [l, r]] = sc[:, :, [r, l]]
        pd['scores'] = sc

    return imgs_flip, pd


def forward_with_flip(model, img, pose_dict, camids=None, target_view=None,
                     use_pose=True, flip_test=True):
    """Forward once or twice (orig + flipped), return averaged feature."""
    kwargs = {}
    if camids is not None:
        kwargs['cam_label'] = camids
    if target_view is not None:
        kwargs['view_label'] = target_view
    if use_pose:
        kwargs['pose_dict'] = pose_dict

    feat, extra = model(img, **kwargs)

    if not flip_test:
        return feat, extra

    img_f, pose_f = flip_batch(img, pose_dict if use_pose else None)
    if use_pose:
        kwargs['pose_dict'] = pose_f
    feat_f, _ = model(img_f, **kwargs)

    # Simple average of concatenated global+part features.
    # Evaluator handles L2-norm downstream.
    return (feat + feat_f) / 2.0, extra
