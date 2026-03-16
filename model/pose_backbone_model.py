"""Pose-guided ReID model with backbone-internal pose injection.

Instead of post-hoc part pooling, injects pose information directly into
the backbone's feature extraction process via Pose Spatial Gates (PSG)
applied between Stage 3 blocks.

This changes HOW features are formed, not just how they're pooled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .make_model import build_transformer
from .modules.pose_spatial_gate import PoseSpatialGate, ContentAdaptivePSG
from .modules.pose_attention_bias import PoseAttentionBias
from .modules.pose_channel_gate import PoseChannelGate
from .modules.pose_cross_attention import PoseCrossAttention
from .modules.pose_reconstruction_head import PoseReconstructionHead
from .modules.pose_utils import merge_person_heatmaps
from .modules.skeleton_gcn import SkeletonGCNHead
from .modules.keypoint_rpe import KeypointRPE, compute_token_kp_distances
from .modules.pose_xcad import PoseCrossAttnHead
from .modules.pose_attn_mask import PoseAttnMask
from .modules.pose_token_decoder import PoseTokenDecoder
from .modules.pose_additive_adapter import PoseAdditiveAdapter, PosePartStructuredAdapter
from .modules.pose_cond_lora import PoseCondLoRA


class PoseBackboneModel(build_transformer):
    """ReID model with pose injection inside backbone.

    Architecture:
    - Swin backbone Stages 0-2: unchanged
    - Stage 3: PSG applied after each SwinBlock
    - Global feature (GAP -> BN -> classifier)
    - No separate part branch

    Test feature = global feature (pose-aware).
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

        # Check injection mode: PSG (post-block gate), PAB (attention bias), PXA (cross-attention), or combo
        self.use_attn_bias = getattr(cfg.MODEL, 'POSE_ATTN_BIAS', False)
        self.use_combo = getattr(cfg.MODEL, 'POSE_PSG_PAB_COMBO', False)
        self.use_cross_attn = getattr(cfg.MODEL, 'POSE_CROSS_ATTN', False)
        self.use_content_adaptive = getattr(cfg.MODEL, 'POSE_PSG_CONTENT_ADAPTIVE', False)
        self.use_attn_mask = getattr(cfg.MODEL, 'POSE_ATTN_MASK', False)
        self.attn_mask_threshold = getattr(cfg.MODEL, 'POSE_ATTN_MASK_THRESHOLD', 0.3)

        # Determine which stages get pose injection
        psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
        num_backbone_stages = len(self.base.stages)
        # Resolve negative indices
        self.psg_stage_indices = set()
        for s in psg_stages:
            idx = s if s >= 0 else num_backbone_stages + s
            self.psg_stage_indices.add(idx)

        hidden_dim = getattr(cfg.MODEL, 'POSE_PFM_HIDDEN', 64)
        spatial_conv = getattr(cfg.MODEL, 'POSE_PSG_SPATIAL', False)

        if self.use_combo:
            # Combo mode: both PAB and PSG
            self.pab_modules_dict = nn.ModuleDict()
            self.psg_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                num_heads = stage.blocks[0].attn.w_msa.num_heads
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.pab_modules_dict[key] = PoseAttentionBias(
                        pose_channels=17,
                        num_heads=num_heads,
                        hidden_dim=32,  # PAB hidden dim
                    )
                    self.psg_modules_dict[key] = PoseSpatialGate(
                        pose_channels=17,
                        feat_channels=feat_ch,
                        hidden_dim=hidden_dim,
                        spatial_conv=spatial_conv,
                    )
        elif self.use_cross_attn:
            # PXA mode: cross-attention between features and pose tokens
            self.pxa_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.pxa_modules_dict[key] = PoseCrossAttention(
                        pose_channels=17,
                        feat_channels=feat_ch,
                        hidden_dim=hidden_dim,
                    )
        elif self.use_attn_bias:
            # PAB-only mode: create PoseAttentionBias modules per stage
            self.pab_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                num_heads = stage.blocks[0].attn.w_msa.num_heads
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.pab_modules_dict[key] = PoseAttentionBias(
                        pose_channels=17,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                    )
        else:
            # PSG-only mode (or CAPSG): create gate modules per stage
            self.psg_modules_dict = nn.ModuleDict()
            GateClass = ContentAdaptivePSG if self.use_content_adaptive else PoseSpatialGate
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    if self.use_content_adaptive:
                        self.psg_modules_dict[key] = GateClass(
                            pose_channels=17,
                            feat_channels=feat_ch,
                            hidden_dim=hidden_dim,
                        )
                    else:
                        self.psg_modules_dict[key] = GateClass(
                            pose_channels=17,
                            feat_channels=feat_ch,
                            hidden_dim=hidden_dim,
                            spatial_conv=spatial_conv,
                        )

            # Backward compatibility: also keep psg_modules list for Stage 3
            # (used by PosePSGPartModel which accesses self.psg_modules)
            last_stage_idx = num_backbone_stages - 1
            if last_stage_idx in self.psg_stage_indices:
                self.psg_modules = nn.ModuleList([
                    self.psg_modules_dict[f's{last_stage_idx}_b{j}']
                    for j in range(len(self.base.stages[last_stage_idx].blocks))
                ])

            # PGAM (Pose-Guided Attention Masking): works alongside PSG
            # PGAM has its own stage config (can differ from PSG stages)
            if self.use_attn_mask:
                pgam_stages = list(getattr(cfg.MODEL, 'POSE_ATTN_MASK_STAGES', [-1]))
                self.pgam_stage_indices = set()
                for s in pgam_stages:
                    idx = s if s >= 0 else num_backbone_stages + s
                    self.pgam_stage_indices.add(idx)
                self.pgam_modules_dict = nn.ModuleDict()
                for stage_idx in sorted(self.pgam_stage_indices):
                    stage = self.base.stages[stage_idx]
                    num_heads = stage.blocks[0].attn.w_msa.num_heads
                    for block_idx in range(len(stage.blocks)):
                        key = f's{stage_idx}_b{block_idx}'
                        self.pgam_modules_dict[key] = PoseAttnMask(
                            num_heads=num_heads,
                            threshold=self.attn_mask_threshold,
                        )

            # PAA (Pose Additive Adapter): additive injection alongside PSG
            self.use_paa = getattr(cfg.MODEL, 'POSE_ADDITIVE_ADAPTER', False)
            self.paa_target_only = getattr(cfg.MODEL, 'POSE_PAA_TARGET_ONLY', False)
            paa_part_structured = getattr(cfg.MODEL, 'POSE_PAA_PART_STRUCTURED', False)
            if self.use_paa:
                self.paa_modules_dict = nn.ModuleDict()
                for stage_idx in sorted(self.psg_stage_indices):
                    stage = self.base.stages[stage_idx]
                    feat_ch = self.base.num_features[stage_idx]
                    for block_idx in range(len(stage.blocks)):
                        key = f's{stage_idx}_b{block_idx}'
                        if paa_part_structured:
                            self.paa_modules_dict[key] = PosePartStructuredAdapter(
                                feat_channels=feat_ch,
                                hidden_per_part=8,
                            )
                        else:
                            paa_routed = getattr(cfg.MODEL, 'POSE_PAA_ROUTED', False)
                            paa_bottleneck = getattr(cfg.MODEL, 'POSE_PAA_BOTTLENECK', 32)
                            self.paa_modules_dict[key] = PoseAdditiveAdapter(
                                pose_channels=17,
                                feat_channels=feat_ch,
                                bottleneck_dim=paa_bottleneck,
                                routed=paa_routed,
                            )
                total_paa_params = sum(p.numel() for p in self.paa_modules_dict.parameters())
                if paa_part_structured:
                    print(f'[PS-PAA] Part-Structured PAA enabled: '
                          f'hidden_per_part=8, total_params={total_paa_params}')
                if self.paa_target_only:
                    print('[S&C] Suppress-and-Complete: PSG=scene, PAA=target(person-0)')

            # PCL (Pose-Conditioned LoRA): feature-dependent alternative to PAA
            self.use_pcl = getattr(cfg.MODEL, 'POSE_COND_LORA', False)
            if self.use_pcl:
                pcl_rank = getattr(cfg.MODEL, 'POSE_COND_LORA_RANK', 16)
                self.pcl_modules_dict = nn.ModuleDict()
                for stage_idx in sorted(self.psg_stage_indices):
                    stage = self.base.stages[stage_idx]
                    feat_ch = self.base.num_features[stage_idx]
                    for block_idx in range(len(stage.blocks)):
                        key = f's{stage_idx}_b{block_idx}'
                        self.pcl_modules_dict[key] = PoseCondLoRA(
                            pose_channels=17,
                            feat_channels=feat_ch,
                            rank=pcl_rank,
                        )
                total_params = sum(p.numel() for p in self.pcl_modules_dict.parameters())
                print(f'[PCL] Pose-Conditioned LoRA enabled: rank={pcl_rank}, '
                      f'total_params={total_params}')

        # Keypoint Relative Position Encoding (KP-RPE)
        self.use_kp_rpe = getattr(cfg.MODEL, 'POSE_KP_RPE', False)
        if self.use_kp_rpe:
            kp_rpe_hidden = getattr(cfg.MODEL, 'POSE_KP_RPE_HIDDEN', 32)
            self.kp_rpe_modules = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                num_heads = stage.blocks[0].attn.w_msa.num_heads
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.kp_rpe_modules[key] = KeypointRPE(
                        num_keypoints=17,
                        num_heads=num_heads,
                        hidden_dim=kp_rpe_hidden,
                    )
            total_params = sum(p.numel() for p in self.kp_rpe_modules.parameters())
            print(f'[KP-RPE] Keypoint Relative Position Encoding enabled: '
                  f'hidden={kp_rpe_hidden}, total_params={total_params}')
            # Store input image size for coordinate mapping
            self._input_h = cfg.INPUT.SIZE_TRAIN[0]
            self._input_w = cfg.INPUT.SIZE_TRAIN[1]

        # Pose-Conditioned Channel Gate (PCG) — after GAP, before BN
        self.use_channel_gate = getattr(cfg.MODEL, 'POSE_CHANNEL_GATE', False)
        if self.use_channel_gate:
            pcg_hidden = getattr(cfg.MODEL, 'POSE_PCG_HIDDEN', 64)
            feat_dim = self.in_planes  # 768 for Swin-Tiny
            self.channel_gate = PoseChannelGate(
                feat_dim=feat_dim,
                pose_channels=17,
                hidden_dim=pcg_hidden,
            )

        # Pose Reconstruction Head (PRA) — auxiliary task
        self.use_recon_head = getattr(cfg.MODEL, 'POSE_RECON_HEAD', False)
        if self.use_recon_head:
            recon_weight = getattr(cfg.MODEL, 'POSE_RECON_WEIGHT', 0.1)
            feat_dim = self.in_planes  # 768 for Swin-Tiny
            self.recon_head = PoseReconstructionHead(
                feat_channels=feat_dim,
                pose_channels=17,
                hidden_channels=128,
                loss_weight=recon_weight,
            )

        # Stochastic Pose Dropout (SPD)
        self.pose_dropout_p = getattr(cfg.MODEL, 'POSE_DROPOUT_P', 0.0)
        if self.pose_dropout_p > 0:
            print(f'[PSG] Stochastic Pose Dropout enabled: p={self.pose_dropout_p}')

        # Pose-Weighted Pooling (PWP) — replace GAP with heatmap-weighted pooling
        self.use_weighted_pool = getattr(cfg.MODEL, 'POSE_WEIGHTED_POOL', False)
        if self.use_weighted_pool:
            print('[PWP] Pose-Weighted Pooling enabled (replaces GAP)')

        # Skeleton GCN head or Cross-Attention Decoder or Pose-Token Decoder (mutually exclusive)
        self.use_skeleton_gcn = getattr(cfg.MODEL, 'POSE_SKELETON_GCN', False)
        self.use_xcad = getattr(cfg.MODEL, 'POSE_XCAD', False)
        self.use_ptd = getattr(cfg.MODEL, 'POSE_TOKEN_DECODER', False)

        if self.use_ptd:
            # Pose-Token Distillation Decoder
            ptd_parts = getattr(cfg.MODEL, 'POSE_TOKEN_NUM_PARTS', 5)
            ptd_dim = getattr(cfg.MODEL, 'POSE_TOKEN_DIM', 256)
            ptd_heads = getattr(cfg.MODEL, 'POSE_TOKEN_HEADS', 8)
            ptd_layers = getattr(cfg.MODEL, 'POSE_TOKEN_LAYERS', 2)
            ptd_hm_weight = getattr(cfg.MODEL, 'POSE_TOKEN_HM_WEIGHT', 1.0)
            self.ptd_decoder = PoseTokenDecoder(
                feat_dim=self.in_planes,
                num_parts=ptd_parts,
                attn_dim=ptd_dim,
                num_heads=ptd_heads,
                num_layers=ptd_layers,
                num_classes=num_classes,
                heatmap_loss_weight=ptd_hm_weight,
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
            total_params = sum(p.numel() for p in self.ptd_decoder.parameters())
            print(f'[PSG+PTD] Pose-Token Decoder enabled: '
                  f'parts={ptd_parts}, dim={ptd_dim}, heads={ptd_heads}, '
                  f'layers={ptd_layers}, hm_weight={ptd_hm_weight}, '
                  f'total_params={total_params}, test_feat={self.pose_test_feat}')
            # Set use_skeleton_gcn=True so existing forward() code path handles part of it
            self.use_skeleton_gcn = True
        elif self.use_xcad:
            # Cross-Attention Decoder replaces GCN
            xcad_dim = getattr(cfg.MODEL, 'POSE_XCAD_DIM', 256)
            xcad_heads = getattr(cfg.MODEL, 'POSE_XCAD_HEADS', 8)
            self.skeleton_head = PoseCrossAttnHead(
                feat_dim=self.in_planes,
                attn_dim=xcad_dim,
                num_heads=xcad_heads,
                num_classes=num_classes,
                input_size=tuple(cfg.INPUT.SIZE_TRAIN),
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
            total_params = sum(p.numel() for p in self.skeleton_head.cross_attn.parameters())
            print(f'[PSG+XCAD] Cross-Attention Decoder enabled: '
                  f'dim={xcad_dim}, heads={xcad_heads}, '
                  f'cross_attn_params={total_params}, '
                  f'test_feat={self.pose_test_feat}')
            # Set use_skeleton_gcn=True so existing forward() code path handles it
            self.use_skeleton_gcn = True
        elif self.use_skeleton_gcn:
            gcn_layers = getattr(cfg.MODEL, 'POSE_GCN_LAYERS', 2)
            gcn_hidden = getattr(cfg.MODEL, 'POSE_GCN_HIDDEN', 256)
            keypoint_pool_only = getattr(cfg.MODEL, 'POSE_KEYPOINT_POOL_ONLY', False)
            kp_weight_mode = getattr(cfg.MODEL, 'POSE_KP_WEIGHT_MODE', 'score')
            kp_triplet = getattr(cfg.MODEL, 'POSE_KP_TRIPLET', False)
            kp_learnable_attn = getattr(cfg.MODEL, 'POSE_KP_LEARNABLE_ATTN', False)
            sgmkc = getattr(cfg.MODEL, 'POSE_SGMKC', False)
            sgmkc_ratio = getattr(cfg.MODEL, 'POSE_SGMKC_RATIO', 0.3)
            kp_uncertainty = getattr(cfg.MODEL, 'POSE_KP_UNCERTAINTY', False)
            kp_uncertainty_reg = getattr(cfg.MODEL, 'POSE_KP_UNCERTAINTY_REG', 0.1)
            pke = getattr(cfg.MODEL, 'POSE_PKE', False)
            self.skeleton_head = SkeletonGCNHead(
                feat_dim=self.in_planes,
                hidden_dim=gcn_hidden,
                num_layers=gcn_layers,
                num_classes=num_classes,
                input_size=tuple(cfg.INPUT.SIZE_TRAIN),
                use_gcn=not keypoint_pool_only,
                kp_weight_mode=kp_weight_mode,
                kp_triplet=kp_triplet,
                kp_learnable_attn=kp_learnable_attn,
                sgmkc=sgmkc,
                sgmkc_ratio=sgmkc_ratio,
                kp_uncertainty=kp_uncertainty,
                kp_uncertainty_reg=kp_uncertainty_reg,
                pke=pke,
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'concat_scaled')
            if keypoint_pool_only:
                print(f'[PSG+KPP] Keypoint pooling head enabled: no graph propagation, '
                      f'test_feat={self.pose_test_feat}, kp_weight={kp_weight_mode}')
            else:
                print(f'[PSG+GCN] Skeleton GCN head enabled: {gcn_layers} layers, '
                      f'hidden={gcn_hidden}, test_feat={self.pose_test_feat}, '
                      f'kp_weight={kp_weight_mode}')

        # PAMC (Pose-Aware Masking Consistency) projector
        self.use_pamc = getattr(cfg.MODEL, 'POSE_PAMC', False)
        if self.use_pamc:
            from .modules.pamc import PAMCProjector, PoseBodyMasker
            proj_dim = getattr(cfg.MODEL, 'POSE_PAMC_PROJ_DIM', 2048)
            self.pamc_projector = PAMCProjector(
                feat_dim=self.in_planes, proj_dim=proj_dim)
            self.pamc_masker = PoseBodyMasker()
            pamc_warmup = getattr(cfg.MODEL, 'POSE_PAMC_WARMUP', 10)
            pamc_weight = getattr(cfg.MODEL, 'POSE_PAMC_WEIGHT', 0.5)
            print(f'[PAMC] Pose-Aware Masking Consistency enabled: '
                  f'proj_dim={proj_dim}, weight={pamc_weight}, '
                  f'warmup={pamc_warmup}')

        # Store backbone's semantic weight for manual forward
        self._semantic_weight_val = semantic_weight

    def _run_backbone_with_psg(self, x, scene_heatmaps, pose_dict=None,
                               paa_heatmaps=None):
        """Run backbone forward with PSG injection in Stage 3.

        Manually iterates backbone stages, inserting PSG after each
        Stage 3 block.

        Args:
            paa_heatmaps: If not None, use these (target-person) heatmaps for PAA
                         instead of scene_heatmaps (Suppress-and-Complete mode).
        """
        # Patch embedding
        x, hw_shape = self.base.patch_embed(x)
        if self.base.use_abs_pos_embed:
            x = x + self.base.absolute_pos_embed
        x = self.base.drop_after_pos(x)

        # Build semantic weight tensor
        sw_val = self._semantic_weight_val
        if self.base.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1, device=x.device) * sw_val
            w = torch.cat([w, 1 - w], dim=-1)
            sem_weight = w
        else:
            sem_weight = None

        outs = []
        num_stages = len(self.base.stages)

        pgam_indices = getattr(self, 'pgam_stage_indices', set())
        for i, stage in enumerate(self.base.stages):
            if i in self.psg_stage_indices or i in pgam_indices:
                # Stage with PSG and/or PGAM: manually run blocks with injection
                x, hw_shape, out, out_hw_shape = self._run_stage_with_psg(
                    stage, x, hw_shape, scene_heatmaps, stage_idx=i,
                    pose_dict=pose_dict, paa_heatmaps=paa_heatmaps)
            else:
                # Normal stage: run without modification
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)

            # Apply semantic weight (from SOLIDER pretraining)
            if sem_weight is not None:
                sw = self.base.semantic_embed_w[i](sem_weight).unsqueeze(1)
                sb = self.base.semantic_embed_b[i](sem_weight).unsqueeze(1)
                x = x * self.base.softplus(sw) + sb

            if i in self.base.out_indices:
                norm_layer = getattr(self.base, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.base.num_features[i]).permute(0, 3, 1,
                                                                   2).contiguous()
                outs.append(out)

        # Pooling
        featmap = outs[-1]  # (B, C, fH, fW)
        if self.use_weighted_pool and scene_heatmaps is not None:
            # Pose-Weighted Pooling: weight tokens by body presence
            fH, fW = featmap.shape[2], featmap.shape[3]
            hm = F.interpolate(scene_heatmaps, size=(fH, fW),
                               mode='bilinear', align_corners=False)
            body_mask = torch.sigmoid(hm).max(dim=1, keepdim=True)[0]  # (B, 1, fH, fW)
            body_mask = body_mask.clamp(min=1e-6)
            global_feat = (featmap * body_mask).sum(dim=(2, 3)) / body_mask.sum(dim=(2, 3))
        else:
            # Standard GAP
            global_feat = self.base.avgpool(featmap)
            global_feat = torch.flatten(global_feat, 1)

        return global_feat, outs

    def _compute_kprpe_bias(self, kp_rpe_module, hw_shape, keypoints, scores,
                            shift_size, window_size):
        """Compute KP-RPE attention bias for a single block.

        Args:
            kp_rpe_module: KeypointRPE module for this block
            hw_shape: (H, W) feature map spatial dimensions
            keypoints: (B, 17, 2) person 0's pixel coordinates
            scores: (B, 17) confidence scores
            shift_size: shift amount for this block (0 or window_size//2)
            window_size: window size (7)

        Returns:
            extra_attn_bias: (B*nW, num_heads, ws*ws, ws*ws)
        """
        H, W = hw_shape
        B = keypoints.shape[0]
        ws = window_size

        # Compute total stride from input to this feature map
        # For Swin-Tiny: patch_size=4, then 3 downsamples of 2x each = 4*2*2*2=32
        stride = self._input_h // H  # should be 32 for 384->12

        # Compute per-token distances to keypoints: (B, H*W, 17)
        token_dists = compute_token_kp_distances(
            hw_shape, keypoints, scores, stride=stride)

        # Reshape to spatial: (B, H, W, 17)
        token_dists = token_dists.view(B, H, W, 17)

        # Pad to multiples of window size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r > 0 or pad_b > 0:
            # Pad with zeros (padded regions have zero distance = no effect)
            token_dists = F.pad(token_dists, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = token_dists.shape[1], token_dists.shape[2]

        # Cyclic shift (for shifted window blocks)
        if shift_size > 0:
            token_dists = torch.roll(
                token_dists,
                shifts=(-shift_size, -shift_size),
                dims=(1, 2))

        # Window partition: (B, H_pad, W_pad, 17)
        #   -> (B, H_pad/ws, ws, W_pad/ws, ws, 17)
        #   -> (B, H_pad/ws, W_pad/ws, ws, ws, 17)
        #   -> (B*nW, ws*ws, 17)
        token_dists = token_dists.view(B, H_pad // ws, ws, W_pad // ws, ws, 17)
        token_dists = token_dists.permute(0, 1, 3, 2, 4, 5).contiguous()
        nW = (H_pad // ws) * (W_pad // ws)
        token_dists = token_dists.view(B * nW, ws * ws, 17)

        # Compute pairwise bias via KeypointRPE
        # Returns: (B*nW, num_heads, ws*ws, ws*ws)
        return kp_rpe_module(token_dists)

    def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps,
                            stage_idx=None, pose_dict=None, paa_heatmaps=None):
        """Run a stage's blocks with pose injection (PSG, PAB, PXA, KP-RPE, or combo)."""
        for block_idx, block in enumerate(stage.blocks):
            key = f's{stage_idx}_b{block_idx}'

            # Compute KP-RPE bias if enabled
            kp_rpe_bias = None
            if self.use_kp_rpe and pose_dict is not None and key in self.kp_rpe_modules:
                keypoints = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
                kp_scores = pose_dict['scores'][:, 0, :]  # (B, 17)
                shift_size = block.attn.shift_size
                window_size = block.attn.window_size
                kp_rpe_bias = self._compute_kprpe_bias(
                    self.kp_rpe_modules[key], hw_shape,
                    keypoints, kp_scores, shift_size, window_size)

            if self.use_combo and scene_heatmaps is not None:
                # Combo mode: PAB inside attention + PSG after block
                pose_bias_map = self.pab_modules_dict[key](
                    scene_heatmaps, hw_shape)
                x = block(x, hw_shape, pose_bias_map=pose_bias_map)
                x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
            elif self.use_cross_attn and scene_heatmaps is not None:
                # PXA mode: run block then apply cross-attention
                x = block(x, hw_shape)
                x = self.pxa_modules_dict[key](x, hw_shape, scene_heatmaps)
            elif self.use_attn_bias and scene_heatmaps is not None:
                # PAB-only mode: compute pose attention bias and pass to block
                pose_bias_map = self.pab_modules_dict[key](
                    scene_heatmaps, hw_shape)
                x = block(x, hw_shape, pose_bias_map=pose_bias_map)
            elif kp_rpe_bias is not None:
                # KP-RPE mode: pass pre-computed bias directly to block
                # Need to pass via extra_attn_bias (bypass pose_bias_map path)
                x = block(x, hw_shape, extra_attn_bias=kp_rpe_bias)
                # PSG still applies after block
                if scene_heatmaps is not None and key in getattr(self, 'psg_modules_dict', {}):
                    x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
            else:
                # PSG-only mode (optionally with PGAM attention masking)
                if self.use_attn_mask and scene_heatmaps is not None and key in getattr(self, 'pgam_modules_dict', {}):
                    # PGAM: generate pose-based attention mask and pass to block
                    pose_bias_map = self.pgam_modules_dict[key](scene_heatmaps, hw_shape)
                    x = block(x, hw_shape, pose_bias_map=pose_bias_map)
                else:
                    x = block(x, hw_shape)
                # PSG: apply gate after block
                if not self.use_attn_bias and scene_heatmaps is not None and key in getattr(self, 'psg_modules_dict', {}):
                    x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
                # PAA: apply additive adapter after PSG
                # S&C mode: use target-person heatmaps for PAA if available
                if getattr(self, 'use_paa', False) and scene_heatmaps is not None and key in getattr(self, 'paa_modules_dict', {}):
                    paa_input = paa_heatmaps if paa_heatmaps is not None else scene_heatmaps
                    x = self.paa_modules_dict[key](x, hw_shape, paa_input)
                # PCL: pose-conditioned LoRA (feature-dependent, replaces PAA)
                if getattr(self, 'use_pcl', False) and scene_heatmaps is not None and key in getattr(self, 'pcl_modules_dict', {}):
                    pcl_input = paa_heatmaps if paa_heatmaps is not None else scene_heatmaps
                    x = self.pcl_modules_dict[key](x, hw_shape, pcl_input)

        # Handle downsample (Stage 3 has no downsample in Swin)
        if stage.downsample:
            x_down, down_hw_shape = stage.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def forward(self, x, label=None, cam_label=None, view_label=None,
                pose_dict=None):
        # Prepare pose
        scene_heatmaps = None
        target_heatmaps = None
        if pose_dict is not None:
            scene_heatmaps, _, target_heatmaps = self._prepare_pose(pose_dict)

        # Stochastic Pose Dropout: zero out heatmaps per-sample during training
        if self.training and scene_heatmaps is not None and self.pose_dropout_p > 0:
            keep_mask = (torch.rand(scene_heatmaps.shape[0], 1, 1, 1,
                                    device=scene_heatmaps.device) >= self.pose_dropout_p)
            scene_heatmaps = scene_heatmaps * keep_mask.float()

        # Run backbone with PSG injection
        # For S&C: pass target_heatmaps to PAA (separate from scene for PSG)
        paa_heatmaps = target_heatmaps if getattr(self, 'paa_target_only', False) else None
        global_feat, featmaps = self._run_backbone_with_psg(
            x, scene_heatmaps,
            pose_dict=pose_dict if self.use_kp_rpe else None,
            paa_heatmaps=paa_heatmaps)

        # Apply Pose-Conditioned Channel Gate (PCG) after GAP, before BN
        if self.use_channel_gate and scene_heatmaps is not None:
            global_feat = self.channel_gate(global_feat, scene_heatmaps)

        # Compute pose reconstruction loss (training only)
        recon_loss = None
        if self.training and self.use_recon_head and scene_heatmaps is not None:
            recon_loss = self.recon_head(featmaps[-1], scene_heatmaps)

        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        feat = self.bottleneck(global_feat)

        if self.training:
            feat_cls = self.dropout(feat)

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            # Part branch: GCN, XCAD, or PTD (detached to prevent gradient interference)
            if self.use_ptd:
                # PTD: use heatmaps for supervision, not as input
                feat_map_detached = featmaps[-1].detach()
                ptd_cls, ptd_feats, ptd_data = self.ptd_decoder(
                    feat_map_detached, scene_heatmaps=scene_heatmaps, return_cls=True)
                # Add heatmap distillation loss to recon_loss
                if 'ptd_heatmap_loss' in ptd_data:
                    if recon_loss is None:
                        recon_loss = ptd_data['ptd_heatmap_loss']
                    else:
                        recon_loss = recon_loss + ptd_data['ptd_heatmap_loss']
                return [cls_score] + ptd_cls, [global_feat] + ptd_feats, featmaps, recon_loss, ptd_data
            elif self.use_skeleton_gcn and pose_dict is not None:
                feat_map_detached = featmaps[-1].detach()
                gcn_cls_scores, gcn_feats, kp_data = self.skeleton_head(
                    feat_map_detached, pose_dict, return_cls=True, label=label)
                # Return lists → triggers list-loss path (implicit 0.5x global)
                # 5th return: kp_data for per-keypoint triplet loss (None if disabled)
                return [cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, featmaps, recon_loss, kp_data

            return cls_score, global_feat, featmaps, recon_loss
        else:
            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            # Part branch test features: PTD (no pose needed) or GCN (needs pose)
            gcn_feats = None
            aux_data = {}
            if self.use_ptd and getattr(self, 'pose_test_feat', 'global') != 'global':
                # PTD: NO pose_dict needed at inference!
                _, gcn_feats, aux_data = self.ptd_decoder(
                    featmaps[-1], scene_heatmaps=None, return_cls=False)
            elif self.use_skeleton_gcn and not self.use_ptd and pose_dict is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                _, gcn_feats, aux_data = self.skeleton_head(
                    featmaps[-1], pose_dict, return_cls=False)

            # Assemble test features from global + part branch (PTD or GCN)
            if gcn_feats is not None:
                if self.pose_test_feat == 'gcn_only':
                    test_feat = torch.cat(gcn_feats, dim=1)
                elif self.pose_test_feat == 'equal_concat':
                    g_norm = F.normalize(test_feat, p=2, dim=1)
                    p_norm = [F.normalize(f, p=2, dim=1) for f in gcn_feats]
                    test_feat = torch.cat([g_norm] + p_norm, dim=1)
                elif self.pose_test_feat in ('cvk_only', 'cvk_hybrid'):
                    test_feat = {
                        'mode': self.pose_test_feat,
                        'global_feat': test_feat,
                        'kp_feats': aux_data.get('kp_feats', gcn_feats[0]),
                        'kp_weights': aux_data.get('kp_weights', torch.ones(1)),
                    }
                else:  # concat_scaled (default)
                    scale = 1.0 / len(gcn_feats)
                    test_feat = torch.cat(
                        [test_feat] + [f * scale for f in gcn_feats], dim=1)

            return test_feat, featmaps

    @staticmethod
    def _prepare_pose(pose_dict):
        """Merge multi-person heatmaps into scene-level tensors.

        Returns:
            scene_heatmaps: (B, 17, H, W) merged scene-level heatmap
            scene_scores: (B, 17) merged confidence scores
            target_heatmaps: (B, 17, H, W) person-0 heatmap (for S&C PAA)
        """
        heatmaps = pose_dict['heatmaps']
        scores = pose_dict['scores']
        person_mask = pose_dict['person_mask']

        scene_heatmaps = merge_person_heatmaps(heatmaps, person_mask)

        score_mask = person_mask.unsqueeze(-1)
        masked_scores = scores * score_mask
        scene_scores = masked_scores.max(dim=1)[0]

        # Target-person (person-0) heatmap for S&C PAA
        # person_mask[:, 0] ensures zero output when no person detected
        target_heatmaps = heatmaps[:, 0] * person_mask[:, 0].view(-1, 1, 1, 1)

        return scene_heatmaps, scene_scores, target_heatmaps
