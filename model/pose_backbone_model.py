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
from .modules.pose_spatial_gate import PoseSpatialGate
from .modules.pose_utils import merge_person_heatmaps
from .modules.skeleton_gcn import SkeletonGCNHead
from .modules.pose_additive_adapter import PoseAdditiveAdapter
from .modules.pair_adaptive_fusion import (
    PairAdaptiveFusionHead,
    PairResidualConfidenceScorer,
    PairResidualScorer,
)


class PoseBackboneModel(build_transformer):
    """ReID model with pose injection inside backbone.

    Architecture:
    - Swin backbone Stages 0-2: unchanged
    - Stage 3: PSG applied after each SwinBlock
    - Global feature (GAP -> BN -> classifier)
    - Optional: Skeleton GCN part branch, PAA adapter, LTCS/LPCS heads

    Test feature = global feature (pose-aware).
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

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

        # PSG-only mode: create gate modules per stage
        self.psg_modules_dict = nn.ModuleDict()
        for stage_idx in sorted(self.psg_stage_indices):
            stage = self.base.stages[stage_idx]
            feat_ch = self.base.num_features[stage_idx]
            for block_idx in range(len(stage.blocks)):
                key = f's{stage_idx}_b{block_idx}'
                self.psg_modules_dict[key] = PoseSpatialGate(
                    pose_channels=17,
                    feat_channels=feat_ch,
                    hidden_dim=hidden_dim,
                    spatial_conv=spatial_conv,
                )

        # Backward compatibility: also keep psg_modules list for Stage 3
        last_stage_idx = num_backbone_stages - 1
        if last_stage_idx in self.psg_stage_indices:
            self.psg_modules = nn.ModuleList([
                self.psg_modules_dict[f's{last_stage_idx}_b{j}']
                for j in range(len(self.base.stages[last_stage_idx].blocks))
            ])

        # PAA (Pose Additive Adapter): additive injection alongside PSG
        self.use_paa = getattr(cfg.MODEL, 'POSE_ADDITIVE_ADAPTER', False)
        if self.use_paa:
            self.paa_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    paa_routed = getattr(cfg.MODEL, 'POSE_PAA_ROUTED', False)
                    paa_bottleneck = getattr(cfg.MODEL, 'POSE_PAA_BOTTLENECK', 32)
                    paa_adaptive_gate = getattr(cfg.MODEL, 'POSE_PAA_ADAPTIVE_GATE', False)
                    self.paa_modules_dict[key] = PoseAdditiveAdapter(
                        pose_channels=17,
                        feat_channels=feat_ch,
                        bottleneck_dim=paa_bottleneck,
                        routed=paa_routed,
                        adaptive_gate=paa_adaptive_gate,
                    )
            total_paa_params = sum(p.numel() for p in self.paa_modules_dict.parameters())
            print(f'[PAA] Pose Additive Adapter enabled: total_params={total_paa_params}')

        # Stochastic Pose Dropout (SPD)
        self.pose_dropout_p = getattr(cfg.MODEL, 'POSE_DROPOUT_P', 0.0)
        if self.pose_dropout_p > 0:
            print(f'[PSG] Stochastic Pose Dropout enabled: p={self.pose_dropout_p}')

        # Skeleton GCN head
        self.use_skeleton_gcn = getattr(cfg.MODEL, 'POSE_SKELETON_GCN', False)
        if self.use_skeleton_gcn:
            gcn_layers = getattr(cfg.MODEL, 'POSE_GCN_LAYERS', 2)
            gcn_hidden = getattr(cfg.MODEL, 'POSE_GCN_HIDDEN', 256)
            keypoint_pool_only = getattr(cfg.MODEL, 'POSE_KEYPOINT_POOL_ONLY', False)
            kp_weight_mode = getattr(cfg.MODEL, 'POSE_KP_WEIGHT_MODE', 'score')
            kp_triplet = getattr(cfg.MODEL, 'POSE_KP_TRIPLET', False)
            self.skeleton_head = SkeletonGCNHead(
                feat_dim=self.in_planes,
                hidden_dim=gcn_hidden,
                num_layers=gcn_layers,
                num_classes=num_classes,
                input_size=tuple(cfg.INPUT.SIZE_TRAIN),
                use_gcn=not keypoint_pool_only,
                kp_weight_mode=kp_weight_mode,
                kp_triplet=kp_triplet,
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'concat_scaled')
            if keypoint_pool_only:
                print(f'[PSG+KPP] Keypoint pooling head enabled: no graph propagation, '
                      f'test_feat={self.pose_test_feat}, kp_weight={kp_weight_mode}')
            else:
                print(f'[PSG+GCN] Skeleton GCN head enabled: {gcn_layers} layers, '
                      f'hidden={gcn_hidden}, test_feat={self.pose_test_feat}, '
                      f'kp_weight={kp_weight_mode}')

        # STD-PR: Structural Token Decomposition (replaces GCN)
        self.use_structural_routing = getattr(cfg.MODEL, 'POSE_STRUCTURAL_ROUTING', False)
        if self.use_structural_routing:
            from .modules.structural_routing import StructuralRoutingLayer
            str_num_parts = getattr(cfg.MODEL, 'POSE_STR_NUM_PARTS', 6)
            str_num_heads = getattr(cfg.MODEL, 'POSE_STR_NUM_HEADS', 8)
            str_num_layers = getattr(cfg.MODEL, 'POSE_STR_NUM_LAYERS', 2)
            self.structural_router = StructuralRoutingLayer(
                feat_dim=self.in_planes,
                num_parts=str_num_parts,
                num_heads=str_num_heads,
                num_layers=str_num_layers,
            )
            # Part classifier for structural tokens
            self.str_classifier = nn.Linear(self.in_planes, num_classes, bias=False)
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
            str_params = sum(p.numel() for p in self.structural_router.parameters())
            print(f'[STD-PR] Structural Token Decomposition enabled: '
                  f'{str_num_parts} parts, {str_num_layers} layers, '
                  f'{str_params} params, test_feat={self.pose_test_feat}')

        # SPLADE: Learned Sparse Projection on GCN pooled feature
        self.use_splade = getattr(cfg.MODEL, 'POSE_SPLADE', False)
        if self.use_splade and self.use_skeleton_gcn:
            from .modules.sparse_head import SparseProjectionHead
            splade_dim = getattr(cfg.MODEL, 'POSE_SPLADE_DIM', 2048)
            self.sparse_head = SparseProjectionHead(
                input_dim=self.in_planes, sparse_dim=splade_dim)
            # Sparse classifier for training the sparse representation
            self.sparse_classifier = nn.Linear(splade_dim, num_classes, bias=False)
            splade_params = sum(p.numel() for p in self.sparse_head.parameters())
            splade_params += sum(p.numel() for p in self.sparse_classifier.parameters())
            print(f'[SPLADE] Sparse projection enabled: dim={splade_dim}, params={splade_params}')

        # LTCS / LPCS heads (pair-adaptive fusion / correction scorer)
        self.use_ltcs = getattr(cfg.MODEL, 'POSE_LTCS', False)
        self.use_lpcs = getattr(cfg.MODEL, 'POSE_LPCS', False)
        if self.use_ltcs and self.use_lpcs:
            raise ValueError('POSE_LTCS and POSE_LPCS cannot be enabled together')
        if self.use_ltcs:
            if not self.use_skeleton_gcn:
                raise ValueError('POSE_LTCS requires POSE_SKELETON_GCN=True')
            ltcs_hidden = getattr(cfg.MODEL, 'POSE_LTCS_HIDDEN', 32)
            self.ltcs_head = PairAdaptiveFusionHead(hidden_dim=ltcs_hidden)
            ltcs_params = sum(p.numel() for p in self.ltcs_head.parameters())
            print(f'[LTCS] Learn-to-Trust Common Support enabled: '
                  f'hidden={ltcs_hidden}, params={ltcs_params}')
        if self.use_lpcs:
            if not self.use_skeleton_gcn:
                raise ValueError('POSE_LPCS requires POSE_SKELETON_GCN=True')
            lpcs_hidden = getattr(cfg.MODEL, 'POSE_LPCS_HIDDEN', 32)
            lpcs_delta_scale = getattr(cfg.MODEL, 'POSE_LPCS_DELTA_SCALE', 0.5)
            lpcs_head_mode = getattr(cfg.MODEL, 'POSE_LPCS_HEAD_MODE', 'residual')
            lpcs_context_mode = getattr(cfg.MODEL, 'POSE_LPCS_CONTEXT_MODE', 'none')
            if lpcs_context_mode in ('query_ctx', 'comp_ctx'):
                lpcs_input_dim = 11
            else:
                lpcs_input_dim = 6
            if lpcs_head_mode == 'residual_conf':
                self.lpcs_head = PairResidualConfidenceScorer(
                    input_dim=lpcs_input_dim,
                    hidden_dim=lpcs_hidden,
                    delta_scale=lpcs_delta_scale,
                )
            else:
                self.lpcs_head = PairResidualScorer(
                    input_dim=lpcs_input_dim,
                    hidden_dim=lpcs_hidden,
                    delta_scale=lpcs_delta_scale,
                )
            lpcs_params = sum(p.numel() for p in self.lpcs_head.parameters())
            print(f'[LPCS] Learned Pair Correction Scorer enabled: '
                  f'head_mode={lpcs_head_mode}, hidden={lpcs_hidden}, delta_scale={lpcs_delta_scale}, '
                  f'context_mode={lpcs_context_mode}, params={lpcs_params}')

        # Store backbone's semantic weight for manual forward
        self._semantic_weight_val = semantic_weight

    def _run_backbone_with_psg(self, x, scene_heatmaps, pose_dict=None):
        """Run backbone forward with PSG injection in configured stages.

        Manually iterates backbone stages, inserting PSG after each block
        in the configured stages.
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
        for i, stage in enumerate(self.base.stages):
            if i in self.psg_stage_indices:
                # Stage with PSG: manually run blocks with injection
                x, hw_shape, out, out_hw_shape = self._run_stage_with_psg(
                    stage, x, hw_shape, scene_heatmaps, stage_idx=i)
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

        # Standard GAP
        global_feat = self.base.avgpool(featmap)
        global_feat = torch.flatten(global_feat, 1)

        return global_feat, outs

    def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps,
                            stage_idx=None):
        """Run a stage's blocks with PSG and optional PAA injection."""
        for block_idx, block in enumerate(stage.blocks):
            key = f's{stage_idx}_b{block_idx}'

            # Run the Swin block
            x = block(x, hw_shape)

            # PSG: apply gate after block
            if scene_heatmaps is not None and key in self.psg_modules_dict:
                x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)

            # PAA: apply additive adapter after PSG
            if getattr(self, 'use_paa', False) and scene_heatmaps is not None and key in getattr(self, 'paa_modules_dict', {}):
                x = self.paa_modules_dict[key](x, hw_shape, scene_heatmaps)

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
        if pose_dict is not None:
            scene_heatmaps, _, _, _ = self._prepare_pose(pose_dict)

        # Stochastic Pose Dropout: zero out heatmaps per-sample during training
        if self.training and scene_heatmaps is not None and self.pose_dropout_p > 0:
            keep_mask = (torch.rand(scene_heatmaps.shape[0], 1, 1, 1,
                                    device=scene_heatmaps.device) >= self.pose_dropout_p)
            scene_heatmaps = scene_heatmaps * keep_mask.float()

        # Run backbone with PSG injection
        global_feat, featmaps = self._run_backbone_with_psg(x, scene_heatmaps)

        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        feat = self.bottleneck(global_feat)

        if self.training:
            feat_cls = self.dropout(feat)
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            # Part branch: STD-PR (structural tokens) or GCN
            if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None:
                feat_map_detached = featmaps[-1].detach()
                B_fm, C_fm, H_fm, W_fm = feat_map_detached.shape
                spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)  # (B, H*W, C)
                structural_tokens, str_stats = self.structural_router(
                    spatial_tokens, (H_fm, W_fm), scene_heatmaps)
                # Part feature: average of structural tokens
                str_feat = structural_tokens.mean(dim=1)  # (B, C)
                # Part classifier
                str_feat_bn = self.structural_router.part_bn(str_feat)
                str_cls = self.str_classifier(str_feat_bn)
                kp_data = {'str_stats': str_stats}
                return [cls_score, str_cls], [global_feat, str_feat], featmaps, None, kp_data

            elif self.use_skeleton_gcn and pose_dict is not None:
                feat_map_detached = featmaps[-1].detach()
                gcn_cls_scores, gcn_feats, kp_data = self.skeleton_head(
                    feat_map_detached, pose_dict, return_cls=True, label=label)

                # SPLADE: auxiliary sparse classification (does NOT modify gcn lists)
                if getattr(self, 'use_splade', False) and len(gcn_feats) > 0:
                    sparse_feat, sparsity = self.sparse_head(gcn_feats[0])
                    sparse_cls = self.sparse_classifier(sparse_feat)
                    if kp_data is None:
                        kp_data = {}
                    kp_data['splade_cls'] = sparse_cls      # separate CE loss in processor
                    kp_data['splade_sparsity'] = sparsity
                    kp_data['splade_reg'] = sparse_feat.mean()  # sparsity regularization

                # Return lists -> triggers list-loss path (implicit 0.5x global)
                return [cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, featmaps, None, kp_data

            return cls_score, global_feat, featmaps, None
        else:
            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            # Part branch test features
            gcn_feats = None
            aux_data = {}
            if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                B_fm, C_fm, H_fm, W_fm = featmaps[-1].shape
                spatial_tokens = featmaps[-1].flatten(2).transpose(1, 2)
                structural_tokens, _ = self.structural_router(
                    spatial_tokens, (H_fm, W_fm), scene_heatmaps)
                str_feat = structural_tokens.mean(dim=1)
                gcn_feats = [str_feat]  # wrap in list for equal_concat compatibility
            elif self.use_skeleton_gcn and pose_dict is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                _, gcn_feats, aux_data = self.skeleton_head(
                    featmaps[-1], pose_dict, return_cls=False)
                # SPLADE: training-only auxiliary, no test-time feature change

            # Assemble test features from global + part branch
            if gcn_feats is not None:
                if self.pose_test_feat == 'gcn_only':
                    test_feat = torch.cat(gcn_feats, dim=1)
                elif self.pose_test_feat == 'equal_concat':
                    g_norm = F.normalize(test_feat, p=2, dim=1)
                    p_norm = [F.normalize(f, p=2, dim=1) for f in gcn_feats]
                    test_feat = torch.cat([g_norm] + p_norm, dim=1)
                elif self.pose_test_feat in ('cvk_only', 'cvk_hybrid', 'cvk_adaptive', 'cvk_residual', 'maxsim', 'maxsim_hybrid'):
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
            target_heatmaps: (B, 17, H, W) person-0 heatmap
            diff_heatmaps: (B, 17, H, W) H_target - H_distractor
        """
        heatmaps = pose_dict['heatmaps']
        scores = pose_dict['scores']
        person_mask = pose_dict['person_mask']

        scene_heatmaps = merge_person_heatmaps(heatmaps, person_mask)

        score_mask = person_mask.unsqueeze(-1)
        masked_scores = scores * score_mask
        scene_scores = masked_scores.max(dim=1)[0]

        # Target-person (person-0) heatmap
        target_heatmaps = heatmaps[:, 0] * person_mask[:, 0].view(-1, 1, 1, 1)

        # Distractor heatmaps: max-merge over non-target persons (indices 1+)
        distractor_mask = person_mask[:, 1:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        distractor_hm = (heatmaps[:, 1:] * distractor_mask).max(dim=1)[0]

        # Differential signal
        diff_heatmaps = target_heatmaps - distractor_hm

        return scene_heatmaps, scene_scores, target_heatmaps, diff_heatmaps
