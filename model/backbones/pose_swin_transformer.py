import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .swin_transformer import SwinTransformer

# -------------------- MMPose predictor (robust) --------------------
_HAS_MMPOSE = False
try:
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmpose.registry import MODELS as MMP_MODELS
    from mmengine.registry import init_default_scope, DefaultScope
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False


class MMPoseTopDownPredictor(nn.Module):
    """Return heatmaps (B,K,h,w) and visibility (B,K)."""
    def __init__(self, cfg_path: str, ckpt_path: str, device: torch.device):
        super().__init__()
        assert _HAS_MMPOSE, "MMPose is required when MODEL.POSE.ENABLE=True"

        # 保证 registry 作用域
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
        # 不走 BaseModel 的 pack 流程，禁用 data_preprocessor
        cfg.model.setdefault('data_preprocessor', None)
        cfg.model['data_preprocessor'] = None

        # 构建模型
        self.model = MMP_MODELS.build(cfg.model).to(device).eval()

        # 形状严格匹配的部分加载（给出加载统计）
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict):
            for k in ['state_dict', 'model', 'state_dict_ema', 'ema', 'module']:
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break
        # 统一常见前缀
        def _strip(k):
            for p in ('model.', 'module.'):
                if k.startswith(p):
                    k = k[len(p):]
            if k.startswith('keypoint_head.'):
                k = 'head.' + k[len('keypoint_head.'):]
            return k
        ckpt = {_strip(k): v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

        msd = self.model.state_dict()
        loadable = {k: v for k, v in ckpt.items() if (k in msd and msd[k].shape == v.shape)}
        missing = [k for k in msd.keys() if k not in loadable]
        unexpected = [k for k in ckpt.keys() if k not in msd]
        self.model.load_state_dict(loadable, strict=False)
        print(f"[PoseSwin][pose_ckpt] loaded={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}")
        assert len(loadable) > 0, "Pose ckpt didn't match any weights."

        # MMPose 期望 0..255，再减均值/除方差
        self.register_buffer('pose_mean', torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('pose_std',  torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 严格反归一化：读取 SOLIDER 的 mean/std
        try:
            from config import cfg as _cfg
            mean = images.new_tensor(_cfg.INPUT.PIXEL_MEAN).view(1,3,1,1)
            std  = images.new_tensor(_cfg.INPUT.PIXEL_STD).view(1,3,1,1)
        except Exception:
            mean = images.new_tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std  = images.new_tensor([0.229,0.224,0.225]).view(1,3,1,1)

        img = images * std + mean      # 0..1
        img = img * 255.0              # 0..255
        img = (img - self.pose_mean) / self.pose_std

        if hasattr(self.model, 'backbone') and hasattr(self.model, 'head'):
            feat = self.model.backbone(img)
            out = self.model.head(feat)
        elif hasattr(self.model, 'extract_feat') and hasattr(self.model, 'head'):
            feat = self.model.extract_feat(img)
            out = self.model.head(feat)
        else:
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


# -------------------- Pose+Swin by composition --------------------
class PoseSwinCompose(nn.Module):
    """
    包含（contain）一个 SwinTransformer，并在指定 stage 融合 pose 热图。
    接口保持与原来一致：forward(x) -> (global_feat, featmaps).
    """
    def __init__(
        self,
        *,
        pretrain_img_size=224,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_cfg=None,                   # 透传给 Swin
        pretrained=None,                 # 供 init_weights 使用
        convert_weights=False,           # 若为 True，可触发 Swin 的转换加载
        semantic_weight=0.0,             # 透传给 Swin（若实现了该特性）
        fusion_mode: str = 'mul',        # 'mul'|'add'|'concat'|'gate'
        fuse_stage: int = 2,             # 0..3
        n_keypoints: int = 17,
        use_visibility: bool = True,
        heatmap_norm: str = 'sigmoid',   # 'none'|'sigmoid'|'softmax'
        pose_predictor: Optional[nn.Module] = None,
        pose_detach: bool = True,
        pose_scale: float = 1.0,
        save_vis: bool = False,
        **kwargs
    ):
        super().__init__()
        assert 0 <= fuse_stage <= 3

        # --- inner Swin ---
        self.swin = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_cfg=init_cfg,
            semantic_weight=semantic_weight,
            **kwargs
        )
        self._pretrained = pretrained
        self._convert_weights = convert_weights

        # --- pose fuse configs ---
        self.fusion_mode = fusion_mode
        self.fuse_stage  = fuse_stage
        self.n_keypoints = n_keypoints
        self.use_visibility = use_visibility
        self.heatmap_norm = heatmap_norm
        self.pose_predictor = pose_predictor
        self.pose_detach = pose_detach
        self.pose_scale = pose_scale
        self.save_vis = save_vis

        # --- 通道配置：out 用 C_i，token x 用 C_{i+1}（最后一层则仍为 C_i） ---
        C_out = self.swin.num_features[fuse_stage]
        C_x   = self.swin.num_features[min(fuse_stage + 1, len(self.swin.num_features) - 1)]

        # proj for out
        self.hm_proj_out = nn.Conv2d(n_keypoints, C_out, kernel_size=1, bias=False)
        # proj for x (after patch-merge, channel doubled)
        self.hm_proj_x   = nn.Conv2d(n_keypoints, C_x,   kernel_size=1, bias=False)

        # （可选）concat/gate 对应层（区分 out 和 x 两条支路）
        if fusion_mode == 'concat':
            self.fuse_proj_out = nn.Conv2d(C_out + C_out, C_out, kernel_size=1, bias=False)
            self.fuse_proj_x   = nn.Conv2d(C_x   + C_x,   C_x,   kernel_size=1, bias=False)
        elif fusion_mode == 'gate':
            self.fuse_gate_out = nn.Conv2d(C_out, C_out, kernel_size=1, bias=True)
            self.fuse_gate_x   = nn.Conv2d(C_x,   C_x,   kernel_size=1, bias=True)

        # 初始化：若想“零影响起步”，可以置零；想一开始就有微扰，可用小随机
        # nn.init.zeros_(self.hm_proj_out.weight)
        # nn.init.zeros_(self.hm_proj_x.weight)
        nn.init.normal_(self.hm_proj_out.weight, std=1e-3)
        nn.init.normal_(self.hm_proj_x.weight,   std=1e-3)
        # 输出需要的属性（与原 Swin 保持一致，便于上层读取）
        self.num_features = self.swin.num_features
        self.avgpool = self.swin.avgpool

        # 缓存
        self.register_buffer('last_hm', torch.zeros(1, n_keypoints, 8, 6), persistent=False)
        self._hm_fullres: Optional[torch.Tensor] = None
        self._vis: Optional[torch.Tensor] = None
        self.pose_enabled = (self.pose_predictor is not None) and (float(self.pose_scale) != 0.0)

    # 只给 Swin 主干加载预训练（不会触碰 pose 分支）
    def init_weights(self, pretrained=None):
        path = pretrained if pretrained is not None else self._pretrained
        if not path:
            self.swin.init_weights(pretrained=None)
            return

        def _pick_state_dict(obj):
            if isinstance(obj, dict):
                for k in ['student', 'state_dict', 'model', 'teacher', 'ema', 'module']:
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k]
                return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            return obj

        def _rename(k: str) -> str:
            if k.startswith('module.'):
                k = k[len('module.'):]
            if k.startswith('backbone.'):
                k = k[len('backbone.'):]
            if k.startswith('layers.'):
                k = 'stages.' + k[len('layers.'):]
            return k

        try:
            obj = torch.load(path, map_location='cpu')
            sd_raw = _pick_state_dict(obj)
            sd = {}
            for k, v in sd_raw.items():
                if not isinstance(v, torch.Tensor):
                    continue
                rk = _rename(k)
                if rk.startswith('head.') or rk.startswith('cls_head.') or rk.startswith('neck.'):
                    continue
                sd[rk] = v

            msd = self.swin.state_dict()
            loadable = {k: v for k, v in sd.items() if (k in msd and msd[k].shape == v.shape)}
            missing = [k for k in msd.keys() if k not in loadable]
            unexpected = [k for k in sd.keys() if k not in msd]
            self.swin.load_state_dict(loadable, strict=False)
            print(f"[PoseSwin][swin_ckpt] remap loaded={len(loadable)} miss={len(missing)} unexp={len(unexpected)} from {path}")

            # 不要再调用 self.swin.init_weights(None) 以免把已加载参数重置
        except Exception as e:
            print(f"[PoseSwin][swin_ckpt] remap failed: {e}; fallback convert_weights={self._convert_weights}")
            if self._convert_weights:
                self.swin.init_weights(path)
            else:
                self.swin.init_weights(pretrained=None)

    # --------- helpers ---------
    def _norm_heatmap(self, hm: torch.Tensor) -> torch.Tensor:
        if self.heatmap_norm == 'none':
            return hm
        if self.heatmap_norm == 'sigmoid':
            return hm.sigmoid()
        if self.heatmap_norm == 'softmax':
            B, K, h, w = hm.shape
            return F.softmax(hm.view(B, K, -1), dim=-1).view(B, K, h, w)
        return hm

    def _fuse_with(self, feat: torch.Tensor, hm: torch.Tensor,
                   proj: nn.Conv2d,
                   fuse_gate: Optional[nn.Module] = None,
                   fuse_proj: Optional[nn.Module] = None) -> torch.Tensor:
        """与给定的投影/门控层配套融合（保证通道对齐）。"""
        hmp = proj(hm)
        if self.fusion_mode == 'mul':
            gate01 = torch.sigmoid(hmp)
            gate = 1.0 + self.pose_scale * (2.0 * gate01 - 1.0)  # scale=1 → [0,2]
            return feat * gate
        elif self.fusion_mode == 'add':
            return feat + hmp * self.pose_scale
        elif self.fusion_mode == 'gate':
            assert fuse_gate is not None
            gate = torch.sigmoid(fuse_gate(hmp))
            return feat * (1.0 + gate * self.pose_scale)
        elif self.fusion_mode == 'concat':
            assert fuse_proj is not None
            fused = fuse_proj(torch.cat([feat, hmp], dim=1))
            return feat + self.pose_scale * (fused - feat)
        else:
            return feat

    @torch.no_grad()
    def _maybe_get_pose_from_images(self, images: torch.Tensor):
        if self.pose_predictor is None:
            self._hm_fullres, self._vis = None, None
            return
        hm, vis = self.pose_predictor(images)
        if self.pose_detach:
            hm = hm.detach()
            vis = vis.detach() if vis is not None else None
        self._hm_fullres, self._vis = hm, vis

    def _resized_pose(self, target_hw, B_expected, device):
        if self._hm_fullres is None:
            h, w = target_hw
            return torch.zeros((B_expected, self.n_keypoints, h, w), device=device)
        hm = F.interpolate(self._hm_fullres, size=target_hw, mode='bilinear', align_corners=False)
        if self.use_visibility and (self._vis is not None):
            hm = hm * self._vis.unsqueeze(-1).unsqueeze(-1).clamp_min(0.0)
        hm = self._norm_heatmap(hm)
        self.last_hm = hm
        return hm

    # --------- forward ---------
    def forward(self, x, semantic_weight=None):
        # 先从输入图像得到一次 pose 热图
        if self.pose_enabled:
            self._maybe_get_pose_from_images(x)
        else:
            self._hm_fullres, self._vis = None, None

        # === patch embedding & pos ===
        x, hw_shape = self.swin.patch_embed(x)
        if getattr(self.swin, 'use_abs_pos_embed', False):
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.swin.stages):
            # stage 返回：给下一层的 tokens（x, hw_shape） 和 当前层输出图 (out, out_hw_shape)
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)  # out: (B, N, C)

            # 在选定 stage 融合 pose
            if i == self.fuse_stage and self.pose_enabled:
                # ---- 1) 对本层输出 out 融合（用于多尺度分支/可视化） ----
                H, W = out_hw_shape
                B, N, C = out.shape
                out_4d = out.transpose(1, 2).contiguous().view(B, C, H, W)
                hm = self._resized_pose((H, W), B_expected=B, device=out_4d.device)

                if self.save_vis and not hasattr(self, '_pose_log_once'):
                    print(f'[PoseSwin] ENABLED | fusion={self.fusion_mode} '
                          f'stage={self.fuse_stage} use_vis={self.use_visibility} '
                          f'norm={self.heatmap_norm} scale={self.pose_scale}')
                    full_shape = None if self._hm_fullres is None else tuple(self._hm_fullres.shape)
                    print(f'[PoseSwin] hm_fullres={full_shape}')
                    self._pose_log_once = True
                if self.save_vis:
                    feat_before = out_4d.detach().abs().mean().item()
                    hm_sum = hm.detach().abs().sum().item()
                    print(f'[PoseSwin] fuse_stage={i} hm_sum={hm_sum:.3f} feat_before_mean={feat_before:.6f}')

                out_4d = self._fuse_with(
                    out_4d, hm,
                    proj=self.hm_proj_out,
                    fuse_gate=getattr(self, 'fuse_gate_out', None),
                    fuse_proj=getattr(self, 'fuse_proj_out', None)
                )

                if self.save_vis:
                    feat_after = out_4d.detach().abs().mean().item()
                    print(f'[PoseSwin] fuse_stage={i} feat_after_mean={feat_after:.6f}')

                out = out_4d.flatten(2).transpose(1, 2).contiguous()

                # ---- 2) 对将要流向下一 stage 的 tokens x 也做同样融合并回写 ----
                Hx, Wx = hw_shape
                B2, N2, C2 = x.shape
                assert N2 == Hx * Wx, "Token 数与 hw_shape 不一致"
                x_4d = x.transpose(1, 2).contiguous().view(B2, C2, Hx, Wx)
                hm_x = self._resized_pose((Hx, Wx), B_expected=B2, device=x_4d.device)
                x_4d = self._fuse_with(
                    x_4d, hm_x,
                    proj=self.hm_proj_x,
                    fuse_gate=getattr(self, 'fuse_gate_x', None),
                    fuse_proj=getattr(self, 'fuse_proj_x', None)
                )
                x = x_4d.flatten(2).transpose(1, 2).contiguous()

            elif i == self.fuse_stage and self.save_vis and not hasattr(self, '_pose_skip_log'):
                print('[PoseSwin] pose disabled → skip fusion')
                self._pose_skip_log = True

            # 语义缩放（若 Swin 支持 & 传入了 semantic_weight）
            if hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0 and (semantic_weight is not None):
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb

            # 收集需要的输出尺度
            if i in self.swin.out_indices:
                norm_layer = getattr(self.swin, f'norm{i}', None)
                if norm_layer is not None:
                    out = norm_layer(out)  # 这里的 out 仍是 (B,N,C)
                B, N, C = out.shape
                H, W = out_hw_shape
                out = out.transpose(1, 2).contiguous().view(B, C, H, W)
                outs.append(out)

        x = self.swin.avgpool(outs[-1])
        x = torch.flatten(x, 1)
        return x, outs


# -------------------- constructors (factory) --------------------
def pose_swin_base_patch4_window7_224(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                                      drop_path_rate=0., pretrained=None, convert_weights=False,
                                      semantic_weight=0.0, pose_cfg=None, **kwargs):
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

    return PoseSwinCompose(
        pretrain_img_size=img_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) if (pretrained and convert_weights) else None,
        pretrained=pretrained,
        convert_weights=convert_weights,
        semantic_weight=semantic_weight,
        fusion_mode=pose_cfg.get('FUSION_MODE', 'mul') if pose_cfg else 'mul',
        fuse_stage=pose_cfg.get('FUSE_STAGE', 2) if pose_cfg else 2,
        n_keypoints=pose_cfg.get('N_KPTS', 17) if pose_cfg else 17,
        use_visibility=pose_cfg.get('USE_VIS', True) if pose_cfg else True,
        heatmap_norm=pose_cfg.get('HM_NORM', 'sigmoid') if pose_cfg else 'sigmoid',
        pose_predictor=pose_predictor,
        pose_detach=pose_cfg.get('DETACH', True) if pose_cfg else True,
        pose_scale=pose_cfg.get('SCALE', 1.0) if pose_cfg else 1.0,
        save_vis=pose_cfg.get('SAVE_VIS', False) if pose_cfg else False,
        **kwargs
    )

def pose_swin_small_patch4_window7_224(*args, **kwargs):
    return pose_swin_base_patch4_window7_224(*args, **kwargs)

def pose_swin_tiny_patch4_window7_224(*args, **kwargs):
    return pose_swin_base_patch4_window7_224(*args, **kwargs)
