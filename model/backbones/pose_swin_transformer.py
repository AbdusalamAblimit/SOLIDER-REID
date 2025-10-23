import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import copy

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
    """Swin backbone with an auxiliary pose-guided local branch.

    The network shares patch embedding and the first ``branch_stage`` Swin stages
    between the global and local branches. The shared feature map is routed to the
    global branch unchanged, while a pose heatmap is fused before feeding the
    local branch. Stages after the branching point are deep-copied so that both
    branches keep identical architectures and can leverage ImageNet pre-training.
    """

    def __init__(
        self,
        *,
        pretrain_img_size=224,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_cfg=None,
        pretrained=None,
        convert_weights=False,
        semantic_weight=0.0,
        fusion_mode: str = 'mul',
        fuse_stage: int = 2,
        n_keypoints: int = 17,
        use_visibility: bool = True,
        heatmap_norm: str = 'sigmoid',
        pose_predictor: Optional[nn.Module] = None,
        pose_detach: bool = True,
        pose_scale: float = 1.0,
        save_vis: bool = False,
        branch_stage: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert fusion_mode in {'mul', 'add', 'concat', 'gate'}, f"Unknown fusion mode: {fusion_mode}"

        self.swin = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_cfg=init_cfg,
            semantic_weight=semantic_weight,
            **kwargs,
        )
        self._pretrained = pretrained
        self._convert_weights = convert_weights

        n_stage = len(self.swin.stages)
        self.fusion_mode = fusion_mode
        self.fuse_stage = max(0, min(fuse_stage, n_stage - 1))
        if branch_stage is None:
            branch_stage = self.fuse_stage
        self.branch_stage = max(0, min(branch_stage, n_stage))
        self.n_keypoints = n_keypoints
        self.use_visibility = use_visibility
        self.heatmap_norm = heatmap_norm
        self.pose_predictor = pose_predictor
        self.pose_detach = pose_detach
        self.pose_scale = pose_scale
        self.save_vis = save_vis

        num_features = self.swin.num_features
        shared_idx = max(self.branch_stage - 1, 0)
        token_idx = min(self.branch_stage, len(num_features) - 1)

        self.hm_proj_tokens = nn.Conv2d(n_keypoints, num_features[token_idx], kernel_size=1, bias=False)
        self.hm_proj_shared = None
        if self.branch_stage > 0:
            self.hm_proj_shared = nn.Conv2d(n_keypoints, num_features[shared_idx], kernel_size=1, bias=False)

        if fusion_mode == 'concat':
            self.fuse_tokens_proj = nn.Conv2d(num_features[token_idx] * 2, num_features[token_idx], kernel_size=1, bias=False)
            self.fuse_shared_proj = None
            if self.hm_proj_shared is not None:
                self.fuse_shared_proj = nn.Conv2d(num_features[shared_idx] * 2, num_features[shared_idx], kernel_size=1, bias=False)
        elif fusion_mode == 'gate':
            self.fuse_tokens_gate = nn.Conv2d(num_features[token_idx], num_features[token_idx], kernel_size=1, bias=True)
            self.fuse_shared_gate = None
            if self.hm_proj_shared is not None:
                self.fuse_shared_gate = nn.Conv2d(num_features[shared_idx], num_features[shared_idx], kernel_size=1, bias=True)
        else:
            self.fuse_tokens_proj = None
            self.fuse_shared_proj = None
            self.fuse_tokens_gate = None
            self.fuse_shared_gate = None

        nn.init.normal_(self.hm_proj_tokens.weight, std=1e-3)
        if self.hm_proj_shared is not None:
            nn.init.normal_(self.hm_proj_shared.weight, std=1e-3)
        if hasattr(self, 'fuse_tokens_proj') and self.fuse_tokens_proj is not None:
            nn.init.normal_(self.fuse_tokens_proj.weight, std=1e-3)
        if hasattr(self, 'fuse_shared_proj') and self.fuse_shared_proj is not None:
            nn.init.normal_(self.fuse_shared_proj.weight, std=1e-3)

        self.local_stages = nn.ModuleList()
        for stage in self.swin.stages[self.branch_stage:]:
            self.local_stages.append(copy.deepcopy(stage))

        self.local_norms = nn.ModuleDict()
        for idx in range(self.branch_stage, n_stage):
            norm_layer = getattr(self.swin, f'norm{idx}', None)
            if norm_layer is not None:
                self.local_norms[str(idx)] = copy.deepcopy(norm_layer)

        self.num_features = self.swin.num_features
        self.avgpool = self.swin.avgpool

        self.register_buffer('last_hm', torch.zeros(1, n_keypoints, 8, 6), persistent=False)
        self._hm_fullres: Optional[torch.Tensor] = None
        self._vis: Optional[torch.Tensor] = None
        self.pose_enabled = (self.pose_predictor is not None) and (float(self.pose_scale) != 0.0)
        self._tb_cache = {}
        self._tb_captured = False

    def _sync_local_branch(self):
        if not self.local_stages:
            return
        for offset, global_stage in enumerate(self.swin.stages[self.branch_stage:]):
            self.local_stages[offset].load_state_dict(global_stage.state_dict())
        for idx in range(self.branch_stage, len(self.swin.stages)):
            norm_global = getattr(self.swin, f'norm{idx}', None)
            norm_key = str(idx)
            norm_local = self.local_norms[norm_key] if norm_key in self.local_norms else None
            if norm_global is not None and norm_local is not None:
                norm_local.load_state_dict(norm_global.state_dict())

    # 只给 Swin 主干加载预训练（不会触碰 pose 分支）
    def init_weights(self, pretrained=None):
        path = pretrained if pretrained is not None else self._pretrained
        if not path:
            self.swin.init_weights(pretrained=None)
            self._sync_local_branch()
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
        except Exception as e:
            print(f"[PoseSwin][swin_ckpt] remap failed: {e}; fallback convert_weights={self._convert_weights}")
            if self._convert_weights:
                self.swin.init_weights(path)
            else:
                self.swin.init_weights(pretrained=None)

        self._sync_local_branch()

    def _norm_heatmap(self, hm: torch.Tensor) -> torch.Tensor:
        if self.heatmap_norm == 'none':
            return hm
        if self.heatmap_norm == 'sigmoid':
            if (not hm.is_cuda) and hm.dtype == torch.float16:
                hm = hm.float()
            return hm.sigmoid()
        if self.heatmap_norm == 'softmax':
            B, K, h, w = hm.shape
            return F.softmax(hm.view(B, K, -1), dim=-1).view(B, K, h, w)
        return hm

    def _fuse_with(
        self,
        feat: torch.Tensor,
        hm: torch.Tensor,
        *,
        proj: Optional[nn.Module] = None,
        fuse_gate: Optional[nn.Module] = None,
        fuse_proj: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if proj is not None:
            hmp = proj(hm)
        else:
            hmp = hm
        if self.fusion_mode == 'mul':
            gate01 = torch.sigmoid(hmp)
            gate = 1.0 + self.pose_scale * (2.0 * gate01 - 1.0)
            return feat * gate
        if self.fusion_mode == 'add':
            return feat + hmp * self.pose_scale
        if self.fusion_mode == 'gate':
            assert fuse_gate is not None
            gate_input = fuse_gate(hmp)
            if (not gate_input.is_cuda) and gate_input.dtype == torch.float16:
                gate_input = gate_input.float()
            gate = torch.sigmoid(gate_input)
            return feat * (1.0 + gate * self.pose_scale)
        if self.fusion_mode == 'concat':
            assert fuse_proj is not None
            fused = fuse_proj(torch.cat([feat, hmp], dim=1))
            return feat + self.pose_scale * (fused - feat)
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

    def reset_pose_debug_epoch(self):
        """让本 epoch 的第一个 batch 可以重新抓取 TB 四件套。"""
        self._tb_cache = {}
        self._tb_captured = False

    def tb_dump_pose(self, writer, step: int, tag_prefix: str = "pose"):
        if not self._tb_cache:
            return
        pack = self._tb_cache
        in_feat = pack.get('in_feat')
        fused_feat = pack.get('fused_feat')
        hm = pack.get('hm')
        hmp = pack.get('hm_proj')
        if in_feat is None or fused_feat is None or hm is None:
            return

        def _reduce_1ch(t: torch.Tensor, how: str = 'mean'):
            t = t.float()
            t = t.sum(dim=1, keepdim=True) if how == 'sum' else t.mean(dim=1, keepdim=True)
            t_min = t.amin(dim=(-2, -1), keepdim=True)
            t_max = t.amax(dim=(-2, -1), keepdim=True)
            return (t - t_min) / (t_max - t_min + 1e-6)

        delta = (fused_feat - in_feat).abs()
        gate01 = None
        if self.fusion_mode == 'mul' and hmp is not None:
            if (not hmp.is_cuda) and hmp.dtype == torch.float16:
                hmp = hmp.float()
            gate01 = torch.sigmoid(hmp)

        writer.add_images(f"{tag_prefix}/feat_in", _reduce_1ch(in_feat, 'mean'), step)
        writer.add_images(f"{tag_prefix}/feat_fused", _reduce_1ch(fused_feat, 'mean'), step)
        writer.add_images(f"{tag_prefix}/hm", _reduce_1ch(hm, 'sum'), step)
        if hmp is not None:
            writer.add_images(f"{tag_prefix}/hm_proj", _reduce_1ch(hmp, 'mean'), step)
        writer.add_images(f"{tag_prefix}/feat_delta", _reduce_1ch(delta, 'mean'), step)
        if gate01 is not None:
            writer.add_images(f"{tag_prefix}/gate01", _reduce_1ch(gate01, 'mean'), step)

        scalars = pack.get('scalars', {})
        for name, value in scalars.items():
            writer.add_scalar(name, float(value), step)

    def forward(self, x, semantic_weight=None):
        if self.pose_enabled:
            self._maybe_get_pose_from_images(x)
        else:
            self._hm_fullres, self._vis = None, None

        x_tokens, hw_shape = self.swin.patch_embed(x)
        if getattr(self.swin, 'use_abs_pos_embed', False):
            x_tokens = x_tokens + self.swin.absolute_pos_embed
        x_tokens = self.swin.drop_after_pos(x_tokens)

        outs_global = []
        outs_local = []
        shared_out = None
        shared_out_norm = None
        shared_out_hw = None

        for i, stage in enumerate(self.swin.stages):
            if i >= self.branch_stage:
                break
            x_tokens, hw_shape, out, out_hw_shape = stage(x_tokens, hw_shape)

            norm_layer = getattr(self.swin, f'norm{i}', None)
            out_collect = out if norm_layer is None else norm_layer(out)
            B, N, C = out_collect.shape
            H, W = out_hw_shape
            out_map = out_collect.transpose(1, 2).contiguous().view(B, C, H, W)
            outs_global.append(out_map)
            outs_local.append(out_map)

            shared_out = out
            shared_out_norm = out_collect
            shared_out_hw = out_hw_shape

            if hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0 and (semantic_weight is not None):
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x_tokens = x_tokens * self.swin.softplus(sw) + sb

        x_global = x_tokens
        hw_global = hw_shape
        x_local = x_tokens
        hw_local = hw_shape

        hm_shared = None
        hm_proj_vis = None
        if self.pose_enabled and shared_out is not None and self.hm_proj_shared is not None:
            B_shared = shared_out.shape[0]
            hm_shared = self._resized_pose(shared_out_hw, B_shared, shared_out.device)
            shared_out_map = shared_out_norm.transpose(1, 2).contiguous().view(B_shared, shared_out_norm.shape[2], shared_out_hw[0], shared_out_hw[1])
            fused_shared = self._fuse_with(
                shared_out_map,
                hm_shared,
                proj=self.hm_proj_shared,
                fuse_gate=getattr(self, 'fuse_shared_gate', None),
                fuse_proj=getattr(self, 'fuse_shared_proj', None),
            )
            if outs_local:
                outs_local[-1] = fused_shared
            hm_proj_vis = self.hm_proj_shared(hm_shared)
        else:
            fused_shared = outs_local[-1] if outs_local else None

        if self.pose_enabled:
            B_tokens = x_local.shape[0]
            hm_tokens = self._resized_pose(hw_local, B_tokens, x_local.device)
            x_local_map = x_local.transpose(1, 2).contiguous().view(B_tokens, x_local.shape[2], hw_local[0], hw_local[1])
            x_local_map = self._fuse_with(
                x_local_map,
                hm_tokens,
                proj=self.hm_proj_tokens,
                fuse_gate=getattr(self, 'fuse_tokens_gate', None),
                fuse_proj=getattr(self, 'fuse_tokens_proj', None),
            )
            x_local = x_local_map.flatten(2).transpose(1, 2).contiguous()
        else:
            hm_tokens = None

        if self.save_vis and not self._tb_captured and shared_out_norm is not None and fused_shared is not None and hm_shared is not None:
            with torch.no_grad():
                sl = slice(0, min(fused_shared.size(0), 4))
                self._tb_cache = {
                    'in_feat': shared_out_norm.transpose(1, 2).contiguous().view(shared_out_norm.size(0), shared_out_norm.size(2), shared_out_hw[0], shared_out_hw[1])[sl].detach().cpu(),
                    'hm': hm_shared[sl].detach().cpu(),
                    'hm_proj': hm_proj_vis[sl].detach().cpu() if hm_proj_vis is not None else None,
                    'fused_feat': fused_shared[sl].detach().cpu(),
                    'scalars': {
                        'feat_before_mean': float(shared_out_norm.detach().abs().mean().item()),
                        'feat_after_mean': float(fused_shared.detach().abs().mean().item()),
                        'hm_sum': float(hm_shared.detach().abs().sum().item()),
                    },
                }
                self._tb_captured = True

        for offset, stage in enumerate(self.swin.stages[self.branch_stage:]):
            idx = self.branch_stage + offset
            stage_local = self.local_stages[offset]
            x_global, hw_global, out_g, out_hw_g = stage(x_global, hw_global)
            x_local, hw_local, out_l, out_hw_l = stage_local(x_local, hw_local)

            if hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0 and (semantic_weight is not None):
                sw = self.swin.semantic_embed_w[idx](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[idx](semantic_weight).unsqueeze(1)
                x_global = x_global * self.swin.softplus(sw) + sb
                x_local = x_local * self.swin.softplus(sw) + sb

            if idx in self.swin.out_indices:
                norm_g = getattr(self.swin, f'norm{idx}', None)
                out_g_collect = out_g if norm_g is None else norm_g(out_g)
                B_g, N_g, C_g = out_g_collect.shape
                Hg, Wg = out_hw_g
                outs_global.append(out_g_collect.transpose(1, 2).contiguous().view(B_g, C_g, Hg, Wg))

                norm_key = str(idx)
                norm_l = self.local_norms[norm_key] if norm_key in self.local_norms else None
                out_l_collect = out_l if norm_l is None else norm_l(out_l)
                B_l, N_l, C_l = out_l_collect.shape
                Hl, Wl = out_hw_l
                outs_local.append(out_l_collect.transpose(1, 2).contiguous().view(B_l, C_l, Hl, Wl))

        global_feat = torch.flatten(self.swin.avgpool(outs_global[-1]), 1)
        local_feat = torch.flatten(self.swin.avgpool(outs_local[-1]), 1)
        concat_feat = torch.cat([global_feat, local_feat], dim=1)

        return {
            'global_feat': global_feat,
            'local_feat': local_feat,
            'concat_feat': concat_feat,
            'global_maps': outs_global,
            'local_maps': outs_local,
        }
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
        branch_stage=pose_cfg.get('BRANCH_STAGE', pose_cfg.get('FUSE_STAGE', 2)) if pose_cfg else 2,
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
