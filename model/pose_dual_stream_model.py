"""Pose-guided Dual Stream (PDS) ReID model.

Architecture:
    Stages 0-2: Shared between Global and Part branches
    Stage 3-G:  Global branch with PSG injection -> GAP -> global feat
    Stage 3-P:  Independent Part branch (deep copy) -> Part Pooling -> 5 part feats

Training:
    Global loss (ID + triplet) + Part loss (5× part ID + part triplet)

Test:
    L2-norm concat: global_feat || part_feats

Key innovation: Gradient-isolated dual branches solve the gradient interference
problem found in exp008/013/014 where PSG and Part Pooling shared Stage 3.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .make_model import build_transformer, weights_init_kaiming, weights_init_classifier
from .modules.pose_spatial_gate import PoseSpatialGate
from .modules.pose_part_pooling import PosePartPooling
from .modules.pose_utils import merge_person_heatmaps
from .modules.skeleton_gcn import SkeletonGCNHead


class PoseDualStreamModel(build_transformer):
    """Dual-stream pose-guided ReID model.

    Global branch: Stage 3 with PSG (pose spatial gate)
    Part branch:   Independent Stage 3 copy with pose part pooling
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

        feat_ch = self.in_planes  # 768 for Swin-Tiny
        hidden_dim = getattr(cfg.MODEL, 'POSE_PFM_HIDDEN', 64)
        last_stage_idx = len(self.base.stages) - 1

        # === Global branch: PSG gates for Stage 3 ===
        self.psg_modules = nn.ModuleList()
        stage3 = self.base.stages[last_stage_idx]
        for _ in range(len(stage3.blocks)):
            self.psg_modules.append(PoseSpatialGate(
                pose_channels=17,
                feat_channels=feat_ch,
                hidden_dim=hidden_dim,
            ))

        # === Part branch: Independent Stage 3 copy ===
        self.part_stage3 = copy.deepcopy(stage3)
        # Independent norm layer for part branch
        self.part_norm3 = copy.deepcopy(getattr(self.base, f'norm{last_stage_idx}'))

        # Part head: Skeleton GCN or Part Pooling
        self.use_skeleton_gcn = getattr(cfg.MODEL, 'POSE_SKELETON_GCN', False)
        if self.use_skeleton_gcn:
            gcn_layers = getattr(cfg.MODEL, 'POSE_GCN_LAYERS', 2)
            gcn_hidden = getattr(cfg.MODEL, 'POSE_GCN_HIDDEN', 256)
            keypoint_pool_only = getattr(cfg.MODEL, 'POSE_KEYPOINT_POOL_ONLY', False)
            input_size = (cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
            kp_weight_mode = getattr(cfg.MODEL, 'POSE_KP_WEIGHT_MODE', 'score')
            kp_triplet = getattr(cfg.MODEL, 'POSE_KP_TRIPLET', False)
            self.skeleton_head = SkeletonGCNHead(
                feat_dim=feat_ch,
                hidden_dim=gcn_hidden,
                num_layers=gcn_layers,
                num_classes=num_classes,
                input_size=input_size,
                use_gcn=not keypoint_pool_only,
                kp_weight_mode=kp_weight_mode,
                kp_triplet=kp_triplet,
            )
        else:
            self.part_pooling = PosePartPooling(
                in_channels=feat_ch,
                num_classes=num_classes,
                threshold=cfg.MODEL.POSE_THRESHOLD,
                heatmap_norm=cfg.MODEL.POSE_HEATMAP_NORM,
                temperature=cfg.MODEL.POSE_TEMPERATURE,
            )

        # Store config
        self._semantic_weight_val = semantic_weight
        self._last_stage_idx = last_stage_idx
        self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
        self.part_stop_grad = getattr(cfg.MODEL, 'POSE_PART_STOP_GRAD', False)
        self.stop_grad_epochs = getattr(cfg.MODEL, 'POSE_STOP_GRAD_EPOCHS', 0)
        self.use_global_psg = getattr(cfg.MODEL, 'POSE_GLOBAL_PSG', True)
        self.current_epoch = 0  # set by processor each epoch

        psg_str = 'PSG' if self.use_global_psg else 'no PSG (ablation)'
        print(f'[PDS] Global branch: Stage 3 ({len(stage3.blocks)} blocks) + {psg_str}')
        if self.use_skeleton_gcn:
            if keypoint_pool_only:
                print('[PDS] Part branch: Independent Stage 3 copy + keypoint pooling only (no graph propagation)')
            else:
                print(f'[PDS] Part branch: Independent Stage 3 copy + Skeleton GCN ({gcn_layers} layers, hidden={gcn_hidden})')
        else:
            print(f'[PDS] Part branch: Independent Stage 3 copy + 5-part pooling')
        if self.stop_grad_epochs > 0:
            print(f'[PDS] Part stop_grad: delayed (block first {self.stop_grad_epochs} epochs, then release)')
        else:
            print(f'[PDS] Part stop_grad: {self.part_stop_grad}')
        print(f'[PDS] Test feature: {self.pose_test_feat}')

    def _run_shared_stages(self, x, scene_heatmaps):
        """Run Stages 0 to N-1 (shared between both branches).

        Returns the token features ready for Stage 3 input,
        plus intermediate feature maps.
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
        last = self._last_stage_idx

        for i, stage in enumerate(self.base.stages):
            if i == last:
                # Don't run Stage 3 here — handled by branches
                break

            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)

            # Apply semantic weight
            if sem_weight is not None:
                sw = self.base.semantic_embed_w[i](sem_weight).unsqueeze(1)
                sb = self.base.semantic_embed_b[i](sem_weight).unsqueeze(1)
                x = x * self.base.softplus(sw) + sb

            if i in self.base.out_indices:
                norm_layer = getattr(self.base, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.base.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return x, hw_shape, sem_weight, outs

    def _run_global_branch(self, x, hw_shape, sem_weight, scene_heatmaps):
        """Global branch: Stage 3 with PSG -> GAP -> global feat."""
        stage3 = self.base.stages[self._last_stage_idx]
        last = self._last_stage_idx

        # Run Stage 3 blocks with PSG (if enabled)
        for block_idx, block in enumerate(stage3.blocks):
            x = block(x, hw_shape)
            if self.use_global_psg and scene_heatmaps is not None:
                x = self.psg_modules[block_idx](x, hw_shape, scene_heatmaps)

        # Stage 3 has no downsample
        out = x
        out_hw_shape = hw_shape

        # Apply semantic weight
        if sem_weight is not None:
            sw = self.base.semantic_embed_w[last](sem_weight).unsqueeze(1)
            sb = self.base.semantic_embed_b[last](sem_weight).unsqueeze(1)
            x = x * self.base.softplus(sw) + sb

        # Norm and reshape to spatial format
        norm_layer = getattr(self.base, f'norm{last}')
        out = norm_layer(out)
        featmap = out.view(-1, *out_hw_shape,
                           self.base.num_features[last]).permute(0, 3, 1, 2).contiguous()

        # GAP
        global_feat = self.base.avgpool(featmap)
        global_feat = torch.flatten(global_feat, 1)

        return global_feat, featmap

    def _run_part_branch(self, x, hw_shape, sem_weight, scene_heatmaps, scene_scores,
                         pose_dict=None, return_cls=True, label=None):
        """Part branch: Independent Stage 3 -> Part Pooling or Skeleton GCN -> part feats."""
        last = self._last_stage_idx

        # Run independent Stage 3 blocks (no PSG)
        for block in self.part_stage3.blocks:
            x = block(x, hw_shape)

        out = x
        out_hw_shape = hw_shape

        # Apply semantic weight (same frozen transform)
        if sem_weight is not None:
            sw = self.base.semantic_embed_w[last](sem_weight).unsqueeze(1)
            sb = self.base.semantic_embed_b[last](sem_weight).unsqueeze(1)
            x = x * self.base.softplus(sw) + sb

        # Norm (independent from global branch)
        out = self.part_norm3(out)
        featmap = out.view(-1, *out_hw_shape,
                           self.base.num_features[last]).permute(0, 3, 1, 2).contiguous()

        if self.use_skeleton_gcn:
            # Skeleton GCN: sample keypoints from feature map -> GCN -> weighted avg
            return self.skeleton_head(featmap, pose_dict, return_cls=return_cls, label=label)
        else:
            # Part pooling
            part_cls_scores, part_feats, part_valid = self.part_pooling(
                featmap, scene_heatmaps, scene_scores)
            return part_cls_scores, part_feats, part_valid

    def forward(self, x, label=None, cam_label=None, view_label=None,
                pose_dict=None):
        # Prepare pose
        scene_heatmaps = None
        scene_scores = None
        if pose_dict is not None:
            scene_heatmaps, scene_scores = self._prepare_pose(pose_dict)

        # Shared stages 0 to N-1
        shared_x, hw_shape, sem_weight, shared_outs = self._run_shared_stages(
            x, scene_heatmaps)

        # Fork: both branches start from same shared tokens
        # Part branch input: detach if stop_grad enabled
        # Delayed stop_grad: block for first N epochs, then release
        if self.stop_grad_epochs > 0:
            should_detach = self.training and (self.current_epoch <= self.stop_grad_epochs)
        else:
            should_detach = self.part_stop_grad

        if should_detach:
            part_input = shared_x.detach()
        else:
            part_input = shared_x.clone()

        # Global branch
        global_feat, global_featmap = self._run_global_branch(
            shared_x.clone(), hw_shape, sem_weight, scene_heatmaps)
        shared_outs.append(global_featmap)

        if self.training:
            # BN + classifier for global
            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)
            feat = self.bottleneck(global_feat)
            feat_cls = self.dropout(feat)
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            # Part branch
            if pose_dict is not None:
                part_cls_scores, part_feats, part_valid = self._run_part_branch(
                    part_input, hw_shape, sem_weight, scene_heatmaps, scene_scores,
                    pose_dict=pose_dict, return_cls=True, label=label)

                all_cls = [cls_score] + part_cls_scores
                all_feats = [global_feat] + part_feats
                return all_cls, all_feats, shared_outs, None
            else:
                return cls_score, global_feat, shared_outs, None
        else:
            # Test: assemble features
            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)
            feat = self.bottleneck(global_feat)

            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            if pose_dict is not None and self.pose_test_feat != 'global':
                _, part_feats, _ = self._run_part_branch(
                    part_input, hw_shape, sem_weight, scene_heatmaps, scene_scores,
                    pose_dict=pose_dict, return_cls=False)

                if self.pose_test_feat == 'part_only':
                    test_feat = torch.cat(part_feats, dim=1)
                elif self.pose_test_feat == 'equal_concat':
                    # L2-normalize each component then concatenate
                    g_norm = F.normalize(test_feat, p=2, dim=1)
                    p_norm = [F.normalize(f, p=2, dim=1) for f in part_feats]
                    test_feat = torch.cat([g_norm] + p_norm, dim=1)
                else:  # concat_scaled
                    scale = 1.0 / len(part_feats)
                    test_feat = torch.cat(
                        [test_feat] + [f * scale for f in part_feats], dim=1)

            return test_feat, shared_outs

    @staticmethod
    def _prepare_pose(pose_dict):
        """Merge multi-person heatmaps into scene-level tensors."""
        heatmaps = pose_dict['heatmaps']
        scores = pose_dict['scores']
        person_mask = pose_dict['person_mask']

        scene_heatmaps = merge_person_heatmaps(heatmaps, person_mask)

        score_mask = person_mask.unsqueeze(-1)
        masked_scores = scores * score_mask
        scene_scores = masked_scores.max(dim=1)[0]

        return scene_heatmaps, scene_scores
