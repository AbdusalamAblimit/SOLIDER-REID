# SOLIDER-REID/model/backbones/pose_swin_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .swin_transformer import SwinTransformer

# ---- Optional MMPose predictor (lightweight wrapper) -----------------
_HAS_MMPOSE = False
try:
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    # 关键：使用 mmpose 自己的 MODELS registry 来构建
    from mmpose.registry import MODELS as MMP_MODELS
    # 关键：设定默认作用域为 mmpose，避免构造 BaseModel 时在 mmengine::model 里找不到
    from mmengine.registry import init_default_scope, DefaultScope
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False


class MMPoseTopDownPredictor(nn.Module):
    """
    Turn a batch of RGB images (B,3,H,W) into keypoint heatmaps (B,K,h,w)
    and visibility (B,K). Expect input already normalized for SOLIDER.
    We re-normalize to MMPose defaults internally.
    """
    def __init__(self, cfg_path: str, ckpt_path: str, device: torch.device):
        super().__init__()
        assert _HAS_MMPOSE, "MMPose is required when MODEL.POSE.ENABLE=True"

        # ---- 保证默认作用域是 mmpose（避免 PoseDataPreprocessor 注册域不匹配） ----
        try:
            cur = DefaultScope.get_current_instance()
            if (cur is None) or (cur.scope != 'mmpose'):
                init_default_scope('mmpose')
        except Exception:
            try:
                init_default_scope('mmpose')
            except Exception:
                pass

        cfg = Config.fromfile(cfg_path)

        # ---- 直接绕过 data_preprocessor 的构建（我们不需要它） ----
        # 有些版本 cfg.model 会包含 data_preprocessor 且类型是 PoseDataPreprocessor，
        # 但注册域不匹配就会在 BaseModel 里报错，这里直接置 None。
        cfg.model.setdefault('data_preprocessor', None)
        cfg.model['data_preprocessor'] = None

        # 用 mmpose 的 MODELS registry 构建
        self.model = MMP_MODELS.build(cfg.model)
        load_checkpoint(self.model, ckpt_path, map_location='cpu')
        self.model.to(device)
        self.model.eval()

        # MMPose normalization expects 0..255 scale then mean/std below
        self.register_buffer('pose_mean',
                             torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('pose_std',
                             torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Tensor[B,3,H,W] RGB. Typically SOLIDER 已做了 ImageNet 归一化。
        Returns:
            heatmap: Tensor[B,K,h,w]
            visibility: Tensor[B,K]
        """
        # 近似把归一化后的 tensor 拉回 0..255，再按 MMPose 规范标准化
        img = images.clamp_(-4.0, 4.0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # 0..1
        img = img * 255.0
        img = (img - self.pose_mean) / self.pose_std

        # 直接走 backbone/head
        if hasattr(self.model, 'backbone') and hasattr(self.model, 'head'):
            feat = self.model.backbone(img)
            out = self.model.head(feat)
        elif hasattr(self.model, 'extract_feat') and hasattr(self.model, 'head'):
            feat = self.model.extract_feat(img)
            out = self.model.head(feat)
        else:
            # 兜底：有些版本 forward 返回 tensor
            out = self.model(img, mode='tensor')

        if isinstance(out, (list, tuple)):
            heatmap, visibility = out[0], (out[1] if len(out) > 1 else None)
        else:
            heatmap, visibility = out, None

        if visibility is None:
            B, K, h, w = heatmap.shape
            visibility = heatmap.view(B, K, -1).amax(dim=-1)
            visibility = (visibility - visibility.min()) / (visibility.max() - visibility.min() + 1e-6)
        return heatmap, visibility


# ---- Pose-aware Swin --------------------------------------------------
class PoseSwinTransformer(SwinTransformer):
    """
    SwinTransformer that fuses precomputed keypoint heatmaps at a chosen stage.
    Keep the same return: forward(x) -> (global_feat, featmaps_list)
    """
    def __init__(
        self,
        *,
        fusion_mode: str = 'mul',          # 'mul' | 'add' | 'concat' | 'gate'
        fuse_stage: int = 2,               # 0..3
        n_keypoints: int = 17,
        use_visibility: bool = True,
        heatmap_norm: str = 'sigmoid',     # 'none' | 'sigmoid' | 'softmax'
        pose_predictor: Optional[nn.Module] = None,
        pose_detach: bool = True,
        pose_scale: float = 1.0,
        save_vis: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 0 <= fuse_stage <= 3
        self.fusion_mode = fusion_mode
        self.fuse_stage  = fuse_stage
        self.n_keypoints = n_keypoints
        self.use_visibility = use_visibility
        self.heatmap_norm = heatmap_norm
        self.pose_predictor = pose_predictor
        self.pose_detach = pose_detach
        self.pose_scale = pose_scale
        self.save_vis = save_vis

        # 每个 stage 的通道数（来自父类）
        C = self.num_features[fuse_stage]
        self.hm_proj = nn.Conv2d(n_keypoints, C, kernel_size=1, bias=False)
        if fusion_mode == 'concat':
            self.fuse_proj = nn.Conv2d(C + C, C, kernel_size=1, bias=False)
        elif fusion_mode == 'gate':
            self.fuse_gate = nn.Conv2d(C, C, kernel_size=1, bias=True)

        # 缓存最后一次用于融合的热图（可用于可视化）
        self.register_buffer('last_hm', torch.zeros(1, n_keypoints, 8, 6), persistent=False)
        self._hm_fullres: Optional[torch.Tensor] = None  # (B,K,H0,W0)
        self._vis: Optional[torch.Tensor] = None         # (B,K)

    def _norm_heatmap(self, hm: torch.Tensor) -> torch.Tensor:
        if self.heatmap_norm == 'none':
            return hm
        if self.heatmap_norm == 'sigmoid':
            return hm.sigmoid()
        if self.heatmap_norm == 'softmax':
            B, K, h, w = hm.shape
            return F.softmax(hm.view(B, K, -1), dim=-1).view(B, K, h, w)
        return hm

    def _fuse(self, feat: torch.Tensor, hm: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W)
        hm:   (B, K, H, W) -- 已 resize & 归一化
        """
        if self.fusion_mode in ('mul', 'add', 'gate', 'concat'):
            hmp = self.hm_proj(hm)  # (B, C, H, W)
        if self.fusion_mode == 'mul':
            return feat * torch.sigmoid(hmp * self.pose_scale)
        elif self.fusion_mode == 'add':
            return feat + hmp * self.pose_scale
        elif self.fusion_mode == 'gate':
            gate = torch.sigmoid(self.fuse_gate(hmp))
            return feat * (1.0 + gate * self.pose_scale)
        elif self.fusion_mode == 'concat':
            cat = torch.cat([feat, hmp], dim=1)
            return self.fuse_proj(cat)
        else:
            return feat

    @torch.no_grad()
    def _maybe_get_pose_from_images(self, images: torch.Tensor):
        """
        从原始输入图像生成 full-res 热图与可见性，仅计算一次，后续各 stage 只做 resize。
        """
        if self.pose_predictor is None:
            self._hm_fullres, self._vis = None, None
            return
        hm, vis = self.pose_predictor(images)
        if self.pose_detach:
            hm = hm.detach()
            vis = vis.detach() if vis is not None else None
        self._hm_fullres, self._vis = hm, vis

    def _resized_pose(self, target_hw: Tuple[int, int], B_expected: int, device) -> torch.Tensor:
        """
        将 full-res 热图 resize 到目标 (H,W)，并做 visibility 与归一化处理。
        """
        if self._hm_fullres is None:
            h, w = target_hw
            return torch.zeros((B_expected, self.n_keypoints, h, w), device=device)

        hm = F.interpolate(self._hm_fullres, size=target_hw, mode='bilinear', align_corners=False)
        if self.use_visibility and (self._vis is not None):
            hm = hm * self._vis.unsqueeze(-1).unsqueeze(-1).clamp_min(0.0)
        hm = self._norm_heatmap(hm)
        self.last_hm = hm
        return hm

    def forward(self, x, semantic_weight=None):
        # 先从输入图像得到姿态热图（一次），后面各个 stage 仅做 resize 对齐
        self._maybe_get_pose_from_images(x)

        # === patch embedding & pos ===
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)  # out: (B, N, C)

            # 在选定 stage 融合 pose：需要暂时变回 (B,C,H,W)
            if i == self.fuse_stage:
                H, W = out_hw_shape
                B, N, C = out.shape
                out_4d = out.transpose(1, 2).contiguous().view(B, C, H, W)
                hm = self._resized_pose((H, W), B_expected=B, device=out_4d.device)
                out_4d = self._fuse(out_4d, hm)
                out = out_4d.flatten(2).transpose(1, 2).contiguous()  # 回到 (B,N,C)

            # 语义缩放：仅当传入了 semantic_weight 才启用
            if (self.semantic_weight >= 0) and (semantic_weight is not None):
                sw = self.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.softplus(sw) + sb

            if i in self.out_indices:
                # 先做 norm（out 仍是 (B,N,C)）
                norm_layer = getattr(self, f'norm{i}', None)
                if norm_layer is not None:
                    out = norm_layer(out)
                # 再 reshape 成 (B,C,H,W)
                B, N, C = out.shape
                H, W = out_hw_shape
                out = out.transpose(1, 2).contiguous().view(B, C, H, W)
                outs.append(out)

        x = self.avgpool(outs[-1])
        x = torch.flatten(x, 1)
        return x, outs


# ---- constructors to match factory signature --------------------------
def pose_swin_base_patch4_window7_224(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                                      drop_path_rate=0., pretrained=None, convert_weights=False,
                                      semantic_weight=0.0, pose_cfg=None, **kwargs):
    # 读取全局 cfg（避免修改 make_model 的参数传递）
    if pose_cfg is None:
        try:
            from config import cfg as _cfg
            pose_cfg = _cfg.MODEL.POSE
        except Exception:
            pose_cfg = None

    pose_predictor = None
    if pose_cfg is not None and pose_cfg.get('ENABLE', False):
        assert _HAS_MMPOSE, "Install mmpose to enable pose."
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pose_predictor = MMPoseTopDownPredictor(pose_cfg['CFG'], pose_cfg['CKPT'], device)

    return PoseSwinTransformer(
        pretrain_img_size=img_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) if (pretrained and convert_weights) else None,
        semantic_weight=semantic_weight,
        fusion_mode=pose_cfg.get('FUSION_MODE', 'mul') if pose_cfg else 'mul',
        fuse_stage=pose_cfg.get('FUSE_STAGE', 2) if pose_cfg else 2,
        n_keypoints=pose_cfg.get('N_KPTS', 17) if pose_cfg else 17,
        use_visibility=pose_cfg.get('USE_VIS', True) if pose_cfg else True,
        heatmap_norm=pose_cfg.get('HM_NORM', 'sigmoid') if pose_cfg else 'sigmoid',
        pose_predictor=pose_predictor,
        pose_detach=pose_cfg.get('DETACH', True) if pose_cfg else True,
        pose_scale=pose_cfg.get('SCALE', 1.0) if pose_cfg else 1.0,
        **kwargs
    )

def pose_swin_small_patch4_window7_224(*args, **kwargs):
    return pose_swin_base_patch4_window7_224(*args, **kwargs)

def pose_swin_tiny_patch4_window7_224(*args, **kwargs):
    return pose_swin_base_patch4_window7_224(*args, **kwargs)
