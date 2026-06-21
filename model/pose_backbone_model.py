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

        # PAPE: Pose-Augmented Patch Embedding — inject pose at input level
        self.use_pose_patch_embed = getattr(cfg.MODEL, 'POSE_PATCH_EMBED', False)
        if self.use_pose_patch_embed:
            embed_dim = self.base.patch_embed.embed_dims  # 96 for Swin-Tiny
            pape_ks = int(getattr(cfg.MODEL, 'POSE_PATCH_EMBED_KS', 1))
            pape_pad = pape_ks // 2  # same-padding for odd kernels
            self.pose_patch_embed = nn.Conv2d(
                17, embed_dim, kernel_size=pape_ks, padding=pape_pad, bias=True)
            # Zero-init so model starts from pretrained behavior
            nn.init.zeros_(self.pose_patch_embed.weight)
            nn.init.zeros_(self.pose_patch_embed.bias)
            pape_params = sum(p.numel() for p in self.pose_patch_embed.parameters())
            print(f'[PAPE] Pose-Augmented Patch Embedding enabled: '
                  f'Conv2d(17→{embed_dim}, {pape_ks}x{pape_ks}), {pape_params} params')

        # Pose Prompt Injection (KPR-style: argmax → learnable part embedding → additive)
        self.use_pose_prompt = getattr(cfg.MODEL, 'POSE_PROMPT', False)
        if self.use_pose_prompt:
            embed_dim = self.base.patch_embed.embed_dims
            num_parts = int(getattr(cfg.MODEL, 'POSE_PROMPT_NUM_PARTS', 18))
            self.pose_prompt_num_parts = num_parts
            self.pose_prompt_drop = float(getattr(cfg.MODEL, 'POSE_PROMPT_DROP', 0.0))
            # Learnable part embedding table: [background, kp0, kp1, ..., kp16]
            self.pose_prompt_embed = nn.Embedding(num_parts, embed_dim)
            # KPR uses trunc_normal_(std=0.02), not zero-init
            nn.init.trunc_normal_(self.pose_prompt_embed.weight, std=0.02)
            # Learnable scale: sigmoid(-2)=0.12 → gentle start for pretrained backbone
            self.pose_prompt_scale = nn.Parameter(torch.tensor(-2.0))
            pp_params = num_parts * embed_dim + 1
            print(f'[PosePrompt] KPR-style injection: Embedding({num_parts}, {embed_dim}), '
                  f'{pp_params} params, drop={self.pose_prompt_drop}, '
                  f'init=trunc_normal(0.02), scale=sigmoid(-2.0)=0.12')

        # Stochastic Pose Dropout (SPD)
        self.pose_dropout_p = getattr(cfg.MODEL, 'POSE_DROPOUT_P', 0.0)
        if self.pose_dropout_p > 0:
            print(f'[PSG] Stochastic Pose Dropout enabled: p={self.pose_dropout_p}')

        # Target-only heatmap (multi-person target disambiguation, Occ-PTrack)
        # Default False preserves scene-heatmap (max over all persons) behavior.
        self.use_pose_shuffle = getattr(cfg.MODEL, 'POSE_SHUFFLE', False)
        if self.use_pose_shuffle:
            print('[exp357] POSE_SHUFFLE kill-switch ON: training-only cross-image pose permutation')
        self.use_target_heatmap = getattr(cfg.MODEL, 'POSE_USE_TARGET_HEATMAP', False)
        if self.use_target_heatmap:
            print('[POSE] POSE_USE_TARGET_HEATMAP=True: '
                  'pose modules (PSG/LGPA/VCSR/PPA/STR/FSDC/...) will receive '
                  'person-0 (target) heatmap instead of max-merged scene heatmap.')

        # GSPB: Gradient-Scaled Part Branch
        self._part_grad_scale = float(getattr(cfg.MODEL, 'POSE_PART_GRAD_SCALE', 0.0))
        if self._part_grad_scale > 0:
            print(f'[GSPB] Part branch gradient scale: {self._part_grad_scale}')

        # BA-PKC: Backbone-Aware Per-Keypoint Contrastive
        self.ba_pkc = getattr(cfg.MODEL, 'POSE_BA_PKC', False)
        if self.ba_pkc:
            print('[BA-PKC] Backbone-aware per-keypoint contrastive enabled')

        # BT-PKD: Backbone-Through Per-Keypoint Distillation
        self.bt_pkd = getattr(cfg.MODEL, 'POSE_BT_PKD', False)
        if self.bt_pkd:
            print('[BT-PKD] Backbone-through per-keypoint distillation enabled')

        # VCSR: Visibility-Conditional Semantic Routing (dynamic part gating)
        self.use_vcsr = getattr(cfg.MODEL, 'POSE_VCSR', False)
        if self.use_vcsr:
            from .modules.vcsr_head import VCSRHead
            self.vcsr_head = VCSRHead(
                feat_dim=self.in_planes,
                num_classes=num_classes,
                clip_dim=int(getattr(cfg.MODEL, 'POSE_LGPA_CLIP_DIM', 512)),
                num_heads=int(getattr(cfg.MODEL, 'POSE_LGPA_NUM_HEADS', 8)),
                pose_mask_temp=float(getattr(cfg.MODEL, 'POSE_LGPA_POSE_TEMP', 1.0)),
                vis_threshold=float(getattr(cfg.MODEL, 'POSE_VCSR_VIS_THR', 0.3)),
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')

        # LGPA: Language-Grounded Part Assignment (CLIP + cross-attention + pose)
        self.use_lgpa = getattr(cfg.MODEL, 'POSE_LGPA', False)
        if self.use_lgpa and getattr(cfg.MODEL, 'POSE_PPA', False):
            raise ValueError('POSE_LGPA and POSE_PPA cannot both be enabled')
        if self.use_lgpa:
            from .modules.clip_part_head import CLIPPartHead
            self.clip_part_head = CLIPPartHead(
                feat_dim=self.in_planes,
                num_classes=num_classes,
                clip_dim=int(getattr(cfg.MODEL, 'POSE_LGPA_CLIP_DIM', 512)),
                num_heads=int(getattr(cfg.MODEL, 'POSE_LGPA_NUM_HEADS', 8)),
                pose_mask_temp=float(getattr(cfg.MODEL, 'POSE_LGPA_POSE_TEMP', 1.0)),
            )
            self._lgpa_detach = getattr(cfg.MODEL, 'POSE_LGPA_DETACH', False)
            self._lgpa_no_pose = getattr(cfg.MODEL, 'POSE_LGPA_NO_POSE', False)
            self._lgpa_fixed_bands = getattr(cfg.MODEL, 'POSE_LGPA_FIXED_BANDS', False)
            self._canon_hm_cache = None  # (1,17,H,W) canonical pedestrian pose, built lazily
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
            if self._lgpa_detach:
                print('[LGPA] Running on DETACHED features (no gradient to backbone)')
            if self._lgpa_no_pose:
                print('[LGPA] NO-POSE ablation: heatmaps=None -> no pose-bias/assign/visibility (pure CLIP-text parts)')
            if self._lgpa_fixed_bands:
                print('[LGPA] FIXED-BANDS: per-image pose replaced by a FIXED canonical pedestrian pose '
                      '(fixed CLIP text + fixed anatomical prior, NO per-image pose)')
            if getattr(cfg.MODEL, 'POSE_LGPA_RANDOM_TEXT', False):
                # Attribution ablation: replace CLIP text prototypes with FIXED random unit vectors.
                # If part_only(random) ~= part_only(CLIP), the CLIP semantics contribute ~0 (shell).
                _g = torch.Generator().manual_seed(42)
                _rand = F.normalize(torch.randn(
                    self.clip_part_head.num_labels, self.clip_part_head.clip_dim,
                    generator=_g), p=2, dim=-1)
                with torch.no_grad():
                    self.clip_part_head.clip_text_features.copy_(_rand.float())
                print('[LGPA] RANDOM-TEXT ablation: CLIP text prototypes -> FIXED random vectors (seed 42)')

        # CLIP-ReID-style learnable ID prompts (the WORKING CLIP mechanism, vs dead fixed part text)
        self.use_clip_id_prompt = getattr(cfg.MODEL, 'POSE_CLIP_ID_PROMPT', False)
        if self.use_clip_id_prompt:
            from .modules.clip_id_prompt import CLIPIDPromptLearner
            self.clip_id_prompt = CLIPIDPromptLearner(
                num_classes,
                clip_arch=getattr(cfg.MODEL, 'POSE_CLIP_ID_ARCH', 'ViT-L-14'),
                clip_pretrained=getattr(cfg.MODEL, 'POSE_CLIP_ID_PRETRAINED', 'openai'),
                pose_cond=getattr(cfg.MODEL, 'POSE_CLIP_ID_POSE_PROMPT', False))
            self.clip_id_proj = nn.Linear(self.in_planes, self.clip_id_prompt.clip_dim)
            self.clip_id_temp = float(getattr(cfg.MODEL, 'POSE_CLIP_ID_TEMP', 0.07))
            print(f'[CLIP-ID-Prompt] enabled: proj {self.in_planes}->{self.clip_id_prompt.clip_dim}, temp {self.clip_id_temp}')
            # Option A: pose-guided image feature for the i2t/t2i alignment (pose guides WHAT CLIP aligns)
            self.use_clip_id_pose_guided = getattr(cfg.MODEL, 'POSE_CLIP_ID_POSE_GUIDED', False)
            if self.use_clip_id_pose_guided:
                from .modules.clip_id_prompt import PoseGuidedPool
                _rng = torch.get_rng_state()   # preserve RNG so downstream module inits match exp341
                self.pose_guided_pool = PoseGuidedPool(
                    self.in_planes, float(getattr(cfg.MODEL, 'POSE_CLIP_ID_POSE_TEMP', 1.0)))
                torch.set_rng_state(_rng)
                print('[CLIP-ID-Prompt] POSE-GUIDED (A): i2t/t2i aligns a pose-bias pooled feature, not raw global')
            # Option C: K pose-localized part features, each aligned to the ID prototype
            self.use_clip_id_part_guided = getattr(cfg.MODEL, 'POSE_CLIP_ID_PART_GUIDED', False)
            if self.use_clip_id_part_guided:
                from .modules.clip_id_prompt import PoseGuidedPartPool
                _rng = torch.get_rng_state()   # preserve RNG so downstream module inits match exp341
                self.pose_guided_part_pool = PoseGuidedPartPool(
                    self.in_planes, float(getattr(cfg.MODEL, 'POSE_CLIP_ID_POSE_TEMP', 1.0)))
                torch.set_rng_state(_rng)
                print('[CLIP-ID-Prompt] PART-GUIDED (C): K pose-localized part features aligned to ID prototype')
            # exp347: PARAMETER-FREE de-occluded pooling for the alignment (no params to absorb)
            self.use_clip_id_noparam_pool = getattr(cfg.MODEL, 'POSE_CLIP_ID_NOPARAM_POOL', False)
            if self.use_clip_id_noparam_pool:
                from .modules.clip_id_prompt import PoseWeightedPool
                self.pose_weighted_pool = PoseWeightedPool(float(getattr(cfg.MODEL, 'POSE_CLIP_ID_POSE_TEMP', 4.0)))
                print('[CLIP-ID-Prompt] NOPARAM-POOL (exp347): align DE-OCCLUDED global (param-free) to pure-ID prototype')
                self.use_clip_id_occ_repel = getattr(cfg.MODEL, 'POSE_CLIP_ID_OCC_REPEL', False)
                self.clip_id_occ_repel_w = float(getattr(cfg.MODEL, 'POSE_CLIP_ID_OCC_REPEL_W', 0.5))
                if self.use_clip_id_occ_repel:
                    print('[CLIP-ID-Prompt] OCC-REPEL (exp348): push occluder feature away from ID prototype, w=%.2f' % self.clip_id_occ_repel_w)

            # exp355 PGPD: pose-guided prompt-prototype dark-knowledge distillation (training-only).
            # NOTE: indent 12 — inside `if self.use_clip_id_prompt:`, OUTSIDE the noparam-pool `if`,
            # so it activates for any CLIP-ID config (exp355 does NOT use noparam-pool).
            self.use_pgpd = getattr(cfg.MODEL, 'POSE_PGPD', False)
            if self.use_pgpd:
                assert not getattr(cfg.MODEL, 'POSE_CLIP_ID_POSE_PROMPT', False), \
                    'PGPD assumes per-label-identical ID prototypes (uniq_protos[inv] scatter); ' \
                    'incompatible with POSE_CLIP_ID_POSE_PROMPT (pose-conditioned prompts).'
                self.pgpd_w = float(getattr(cfg.MODEL, 'POSE_PGPD_W', 0.5))
                self.pgpd_tau = float(getattr(cfg.MODEL, 'POSE_PGPD_TAU', 0.1))
                self.pgpd_random_teacher = getattr(cfg.MODEL, 'POSE_PGPD_RANDOM_TEACHER', False)
                print('[CLIP-ID-Prompt] PGPD (exp355): pose-guided prompt dark-distill, w=%.2f tau=%.2f random_teacher=%s'
                      % (self.pgpd_w, self.pgpd_tau, self.pgpd_random_teacher))

            # exp356 PC-MSC: pose-conditioned masked semantic completion (training-only)
            self.use_pcmsc = getattr(cfg.MODEL, 'POSE_PCMSC', False)
            if self.use_pcmsc:
                from .modules.clip_id_prompt import CLIPVisualEncoder
                _rng = torch.get_rng_state()   # preserve RNG so backbone/bottleneck inits match exp341
                self.pcmsc_visual = CLIPVisualEncoder(
                    clip_arch=getattr(cfg.MODEL, 'POSE_CLIP_ID_ARCH', 'ViT-L-14'),
                    clip_pretrained=getattr(cfg.MODEL, 'POSE_CLIP_ID_PRETRAINED', 'openai'))
                self.pcmsc_w = float(getattr(cfg.MODEL, 'POSE_PCMSC_W', 1.0))
                self.pcmsc_random_mask = getattr(cfg.MODEL, 'POSE_PCMSC_RANDOM_MASK', False)
                self.pcmsc_mask_token = nn.Parameter(torch.zeros(self.in_planes))
                self.pcmsc_query = nn.Parameter(torch.randn(3, self.in_planes) * 0.02)  # per-region query
                self.pcmsc_decoder = nn.MultiheadAttention(self.in_planes, num_heads=8, batch_first=True)
                self.pcmsc_proj = nn.Linear(self.in_planes, self.pcmsc_visual.clip_dim)
                torch.set_rng_state(_rng)
                print('[PC-MSC] enabled (exp356): w=%.2f random_mask=%s, decoder %d->%d'
                      % (self.pcmsc_w, self.pcmsc_random_mask, self.in_planes, self.pcmsc_visual.clip_dim))

        # PPA: Pose-Prompted Part-Assignment Head (replaces GCN)
        self.use_ppa = getattr(cfg.MODEL, 'POSE_PPA', False)
        if self.use_ppa:
            from .modules.part_assignment_head import PartAssignmentHead
            self.part_assignment_head = PartAssignmentHead(
                feat_dim=self.in_planes,
                num_classes=num_classes,
                num_parts=int(getattr(cfg.MODEL, 'POSE_PPA_NUM_PARTS', 5)),
                assign_weight=float(getattr(cfg.MODEL, 'POSE_PPA_ASSIGN_WEIGHT', 0.5)),
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')

        # FSDC: Feature-Space Diffusion Completion
        self.use_fsdc = getattr(cfg.MODEL, 'POSE_FSDC', False)
        if self.use_fsdc:
            from .modules.feature_denoiser import FeatureDenoiser
            fH = cfg.INPUT.SIZE_TRAIN[0] // 32  # 384//32 = 12
            fW = cfg.INPUT.SIZE_TRAIN[1] // 32  # 128//32 = 4
            self.feature_denoiser = FeatureDenoiser(
                feat_dim=self.in_planes,
                num_tokens=fH * fW,
                num_layers=int(getattr(cfg.MODEL, 'POSE_FSDC_LAYERS', 2)),
                num_heads=int(getattr(cfg.MODEL, 'POSE_FSDC_HEADS', 8)),
                mask_ratio=float(getattr(cfg.MODEL, 'POSE_FSDC_MASK_RATIO', 0.3)),
                noise_std=float(getattr(cfg.MODEL, 'POSE_FSDC_NOISE_STD', 0.1)),
            )

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
                deformable_sample=getattr(cfg.MODEL, 'POSE_DEFORMABLE_SAMPLE', False),
                deformable_k=int(getattr(cfg.MODEL, 'POSE_DEFORMABLE_K', 4)),
                multi_scale_kp=getattr(cfg.MODEL, 'POSE_MULTI_SCALE_KP', False),
                multi_scale_s2_dim=self.base.num_features[-2] if len(self.base.num_features) >= 2 else self.in_planes,
                per_part=getattr(cfg.MODEL, 'POSE_GCN_PER_PART', False),
                vcnorm=(getattr(cfg.MODEL, 'POSE_VCNORM', False)
                        and getattr(cfg.MODEL, 'POSE_VCNORM_MODULE', True)),
                vcnorm_hidden=int(getattr(cfg.MODEL, 'POSE_VCNORM_HIDDEN', 64)),
                vcnorm_gain_scale=float(getattr(cfg.MODEL, 'POSE_VCNORM_GAIN_SCALE', 1.0)),
            )
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'concat_scaled')
            if keypoint_pool_only:
                print(f'[PSG+KPP] Keypoint pooling head enabled: no graph propagation, '
                      f'test_feat={self.pose_test_feat}, kp_weight={kp_weight_mode}')
            else:
                print(f'[PSG+GCN] Skeleton GCN head enabled: {gcn_layers} layers, '
                      f'hidden={gcn_hidden}, test_feat={self.pose_test_feat}, '
                      f'kp_weight={kp_weight_mode}')

        # VC-Norm: Visibility-Conditioned Normalization on GCN per-keypoint tokens.
        # Treats occlusion as a domain factor (probe: occluded tokens shift their
        # per-channel norm statistics). The VCN affine module is OWNED BY the
        # skeleton_head (created above when POSE_VCNORM_MODULE=True) and applied
        # post-GCN/pre-pool, so it flows into both the pooled ReID feature and the
        # exported kp_feats in BOTH train and test forward paths (symmetry).
        # Zero-init -> identity at start, never breaks baseline reproduction. The
        # batch-level statistic-alignment loss lives in processor.py.
        self.use_vcnorm = getattr(cfg.MODEL, 'POSE_VCNORM', False)
        if self.use_vcnorm and not self.use_skeleton_gcn:
            raise ValueError('POSE_VCNORM requires POSE_SKELETON_GCN=True '
                             '(VC-Norm operates on GCN per-keypoint tokens)')

        # PNIS: Pose-Normalized Identity Space
        self.use_pose_normalize = getattr(cfg.MODEL, 'POSE_NORMALIZE', False)
        if self.use_pose_normalize:
            from .modules.pose_normalizer import PoseNormalizer
            pn_hidden = getattr(cfg.MODEL, 'POSE_NORMALIZE_HIDDEN', 256)
            self.pose_normalizer = PoseNormalizer(
                feat_dim=self.in_planes, hidden_dim=pn_hidden)
            pn_params = sum(p.numel() for p in self.pose_normalizer.parameters())
            print(f'[PNIS] Pose-Normalized Identity Space enabled: {pn_params} params')

        # STD-PR: Structural Token Decomposition (replaces GCN, mutually exclusive)
        self.use_structural_routing = getattr(cfg.MODEL, 'POSE_STRUCTURAL_ROUTING', False)
        # Dual Part Branch: both can now be enabled simultaneously
        # STD-PR provides per-token SupCon, GCN provides architecture via skeleton graph
        if self.use_structural_routing:
            from .modules.structural_routing import StructuralRoutingLayer
            str_num_parts = getattr(cfg.MODEL, 'POSE_STR_NUM_PARTS', 6)
            str_num_heads = getattr(cfg.MODEL, 'POSE_STR_NUM_HEADS', 8)
            str_num_layers = getattr(cfg.MODEL, 'POSE_STR_NUM_LAYERS', 2)
            str_self_attn = getattr(cfg.MODEL, 'POSE_STR_SELF_ATTN', False)
            self.structural_router = StructuralRoutingLayer(
                feat_dim=self.in_planes,
                num_parts=str_num_parts,
                num_heads=str_num_heads,
                num_layers=str_num_layers,
                self_attn=str_self_attn,
            )
            self.str_self_attn = str_self_attn
            # Part classifier for structural tokens
            self.str_classifier = nn.Linear(self.in_planes, num_classes, bias=False)
            self.str_per_token = getattr(cfg.MODEL, 'POSE_STR_PER_TOKEN', False)
            self.str_part_drop = float(getattr(cfg.MODEL, 'POSE_STR_PART_DROP', 0.0))
            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
            str_params = sum(p.numel() for p in self.structural_router.parameters())
            pltd_str = f', part_drop={self.str_part_drop}' if self.str_part_drop > 0 else ''
            print(f'[STD-PR] Structural Token Decomposition enabled: '
                  f'{str_num_parts} parts, {str_num_layers} layers, '
                  f'{str_params} params, test_feat={self.pose_test_feat}{pltd_str}')

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

        # PAPE: add pose patch embedding (early pose injection)
        if getattr(self, 'use_pose_patch_embed', False) and scene_heatmaps is not None:
            H_hw, W_hw = hw_shape
            # Resize heatmaps to match post-PatchEmbed spatial dims
            hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw),
                               mode='bilinear', align_corners=False)
            pose_tokens = self.pose_patch_embed(hm)  # (B, C, H, W)
            pose_tokens = pose_tokens.flatten(2).transpose(1, 2)  # (B, N, C)
            x = x + pose_tokens.to(x.dtype)  # AMP safety

        # Pose Prompt: KPR-style argmax part ID → learnable embedding → additive
        if getattr(self, 'use_pose_prompt', False) and scene_heatmaps is not None:
            H_hw, W_hw = hw_shape
            # Resize 17-channel heatmaps to patch resolution
            hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw),
                               mode='bilinear', align_corners=False)  # (B, 17, H, W)
            # Heatmaps are already [0,1] (ViTPose MSE-trained output, not logits)
            # Only clamp float16 rounding artifacts (tiny negatives)
            hm = hm.clamp(min=0)
            # Background channel: 1 - max keypoint confidence
            bg = 1.0 - hm.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
            hm_with_bg = torch.cat([bg, hm], dim=1)  # (B, 18, H, W)
            # Argmax → part ID per patch (detach: no gradient through heatmaps)
            part_ids = hm_with_bg.detach().argmax(dim=1)  # (B, H, W) values in [0, 17]
            part_ids = part_ids.reshape(part_ids.shape[0], -1)  # (B, N)
            # Stochastic prompt drop during training (use empty prompt = all background)
            if self.training and self.pose_prompt_drop > 0:
                drop_mask = torch.rand(part_ids.shape[0], 1, device=part_ids.device) < self.pose_prompt_drop
                part_ids = torch.where(drop_mask.expand_as(part_ids),
                                       torch.zeros_like(part_ids), part_ids)  # 0 = background
            # Lookup learnable embeddings, scale, and add to patch tokens
            prompt_embeds = self.pose_prompt_embed(part_ids)  # (B, N, C)
            prompt_embeds = prompt_embeds.to(x.dtype)  # AMP safety: match float16/32
            scale = torch.sigmoid(self.pose_prompt_scale)  # learnable injection strength
            x = x + scale * prompt_embeds

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

    def _canonical_heatmap(self, B, device):
        """Fixed canonical upright-pedestrian COCO-17 pose heatmap (NO per-image info).
        FIXED-BANDS mode feeds this in place of per-image pose, giving the CLIP-text part
        queries a fixed anatomical localization prior (head top -> ankles bottom)."""
        if self._canon_hm_cache is None:
            H, W = 96, 32
            KP = [(0.50, 0.06), (0.46, 0.05), (0.54, 0.05), (0.42, 0.06), (0.58, 0.06),
                  (0.36, 0.18), (0.64, 0.18), (0.32, 0.32), (0.68, 0.32), (0.30, 0.45), (0.70, 0.45),
                  (0.40, 0.50), (0.60, 0.50), (0.41, 0.72), (0.59, 0.72), (0.42, 0.95), (0.58, 0.95)]
            ys = torch.arange(H, dtype=torch.float32).view(H, 1)
            xs = torch.arange(W, dtype=torch.float32).view(1, W)
            hm = torch.zeros(1, 17, H, W)
            sx, sy = 0.12 * W, 0.05 * H
            for k, (nx, ny) in enumerate(KP):
                cx, cy = nx * W, ny * H
                hm[0, k] = torch.exp(-(((xs - cx) ** 2) / (2 * sx ** 2) + ((ys - cy) ** 2) / (2 * sy ** 2)))
            self._canon_hm_cache = hm
        return self._canon_hm_cache.to(device).expand(B, 17, -1, -1).contiguous()

    def _pgpd_loss(self, img_proj, txt_proto, label, scene_heatmaps, target_heatmaps=None):
        """exp355 PGPD: pose-guided prompt-prototype dark-knowledge distillation.
        Pose selects a more-complete same-ID teacher within the batch; distill the
        teacher's soft distribution over the batch's OTHER-ID prototypes (hard-negatives)
        to the (occluded) student. Training-only: no test-time pose, no new ID pathway in
        the CLIP alignment, descriptor unchanged. Returns a scalar loss tensor.
        target_heatmaps (target person only) is used for completeness so a distractor in a
        multi-person scene cannot inflate it (Codex Medium)."""
        import torch.nn.functional as F
        B = img_proj.shape[0]
        device = img_proj.device
        # unique ID prototypes within the batch. Same-ID images share an identical prototype
        # because exp355 keeps POSE_CLIP_ID_POSE_PROMPT off (prototype depends only on label),
        # so scattering by inverse index is consistent.
        uniq_labels, inv = torch.unique(label, return_inverse=True)   # inv: (B,) in [0,P)
        P = uniq_labels.shape[0]
        if P < 3:
            return img_proj.new_zeros(())          # need >=2 hard-negatives for a soft target
        uniq_protos = img_proj.new_zeros(P, txt_proto.shape[1], dtype=torch.float32)  # fp32: match txt_proto.float() (AMP index_put dtype)
        uniq_protos[inv] = txt_proto.float()
        # image-to-ID-prototype logits (fp32 for softmax stability under AMP)
        img_n = F.normalize(img_proj.float(), dim=1)
        proto_n = F.normalize(uniq_protos, dim=1)
        logits = img_n @ proto_n.t() / self.pgpd_tau           # (B, P)
        # TARGET pose completeness per image (sum of per-keypoint peak activation), detached.
        # Use target_heatmaps (target person only), NOT scene_heatmaps (max-merged over all
        # persons) — else a distractor can inflate completeness while the target is occluded,
        # driving teacher selection by non-target pose (Codex Medium). Fall back to scene only
        # if target is unavailable.
        comp_hm = target_heatmaps if target_heatmaps is not None else scene_heatmaps
        comp = comp_hm.float().amax(dim=(2, 3)).sum(dim=1).detach()   # (B,)
        same = label.view(B, 1) == label.view(1, B)            # (B, B)
        not_self = ~torch.eye(B, dtype=torch.bool, device=device)
        if self.pgpd_random_teacher:
            cand = same & not_self                             # control: any same-ID teacher
            score = torch.rand(B, B, device=device)            # random pick among candidates
        else:
            cand = same & (comp.view(1, B) > comp.view(B, 1)) & not_self  # teacher strictly more complete
            score = comp.view(1, B).expand(B, B)               # pick the most-complete teacher
        score = torch.where(cand, score, score.new_full((B, B), -1e9))
        teacher_idx = score.argmax(dim=1)                      # (B,)
        has_teacher = cand.any(dim=1)                          # (B,)
        # hard-negative mask: drop each student's true-ID column (= inv). The teacher shares the
        # same label, so the same column is its true ID too.
        neg_mask = torch.ones(B, P, dtype=torch.bool, device=device)
        neg_mask[torch.arange(B, device=device), inv] = False
        student_logp = F.log_softmax(logits.masked_fill(~neg_mask, float('-inf')), dim=1)   # (B,P)
        teacher_p = F.softmax(logits[teacher_idx].masked_fill(~neg_mask, float('-inf')), dim=1).detach()
        # cross-entropy KD over hard-negatives; zero the masked column to avoid 0*(-inf)=NaN
        prod = (teacher_p * student_logp).masked_fill(~neg_mask, 0.0)
        dark = -prod.sum(dim=1)                                # (B,)
        dark = torch.nan_to_num(dark, nan=0.0, posinf=0.0, neginf=0.0)
        # weight: how much more complete the teacher is; 0 for students with no teacher
        if self.pgpd_random_teacher:
            w = has_teacher.float()
        else:
            w = (comp[teacher_idx] - comp).clamp(min=0.0) * has_teacher.float()
        pgpd = (w * dark).sum() / w.sum().clamp(min=1e-6)
        if not getattr(self, '_pgpd_logged', False):
            self._pgpd_logged = True
            print('[PGPD] first-call diag: teacher coverage %d/%d, mean_w %.3f, mean_dark %.3f, P %d'
                  % (int(has_teacher.sum()), B, float(w.mean()), float(dark.mean()), P))
        return self.pgpd_w * pgpd

    def _pcmsc_loss(self, featmap, img, scene_heatmaps):
        """exp356 PC-MSC: pose-masked CLIP-semantic completion. Mask a (visible) region's
        backbone tokens; reconstruct that region's frozen CLIP-visual feature from the visible
        context. Training-only regularizer; the descriptor (global) is computed on the UNMASKED
        featmap elsewhere. Returns a scalar loss tensor."""
        import torch.nn.functional as F
        B, C, H, W = featmap.shape
        device = featmap.device
        target = self.pcmsc_visual.part_targets(img)            # (B,3,clip_dim) frozen, fp32, detached
        # region of each token by row (5/6/5 split of H ~ CLIP 16-grid head/torso/legs thirds)
        h1, h2 = round(5.0 / 16 * H), round(11.0 / 16 * H)
        h1 = max(1, min(h1, H - 2)); h2 = max(h1 + 1, min(h2, H - 1))
        rows = torch.arange(H, device=device).repeat_interleave(W)       # (HW,)
        region_of_token = (rows >= h1).long() + (rows >= h2).long()      # (HW,) in {0,1,2}
        # per-region pose visibility
        pose = F.interpolate(scene_heatmaps.float(), size=(H, W), mode='bilinear', align_corners=False)
        vis_map = pose.amax(dim=1).flatten(1)                            # (B, HW)
        reg_vis = torch.stack([vis_map[:, region_of_token == r].mean(1) for r in range(3)], dim=1)  # (B,3)
        # select region to mask: visibility-weighted (pose) or uniform (control)
        if self.pcmsc_random_mask:
            sel = torch.randint(0, 3, (B,), device=device)
        else:
            sel = torch.multinomial(reg_vis.clamp(min=1e-6).softmax(dim=1), 1).squeeze(1)  # (B,)
        # mask the selected region's tokens with the learnable mask token
        tokens = featmap.flatten(2).transpose(1, 2)                      # (B, HW, C)
        mask = (region_of_token.unsqueeze(0) == sel.unsqueeze(1))        # (B, HW) bool
        mt = self.pcmsc_mask_token.view(1, 1, C).to(tokens.dtype)
        tok_masked = torch.where(mask.unsqueeze(-1), mt, tokens)
        # decoder: the selected region's query reconstructs from the (masked) token set
        q = self.pcmsc_query[sel].unsqueeze(1).to(tokens.dtype)          # (B,1,C)
        R = self.pcmsc_decoder(q, tok_masked, tok_masked)[0].squeeze(1)  # (B, C)
        R = F.normalize(self.pcmsc_proj(R).float(), dim=-1)             # (B, clip_dim) fp32
        tgt = target[torch.arange(B, device=device), sel]               # (B, clip_dim)
        cos = (R * tgt).sum(-1)
        loss = (1.0 - cos).mean()
        if not getattr(self, '_pcmsc_logged', False):
            self._pcmsc_logged = True
            print('[PC-MSC] first-call diag: sel-region hist %s, mean cos %.3f'
                  % (torch.bincount(sel, minlength=3).tolist(), float(cos.mean())))
        return self.pcmsc_w * loss

    def _lgpa_heatmap(self, scene_heatmaps, B, device):
        """Select the heatmap fed to the LGPA head: None (no-pose), fixed canonical
        (fixed-bands), or per-image scene heatmaps (default)."""
        if getattr(self, '_lgpa_no_pose', False):
            return None
        if getattr(self, '_lgpa_fixed_bands', False):
            return self._canonical_heatmap(B, device)
        return scene_heatmaps

    def forward(self, x, label=None, cam_label=None, view_label=None,
                pose_dict=None):
        # Prepare pose
        scene_heatmaps = None
        target_heatmaps = None
        if pose_dict is not None:
            scene_heatmaps, _, target_heatmaps, _ = self._prepare_pose(pose_dict)
            # exp357 pose-shuffle kill-switch: training-only cross-image permutation of the pose
            # within the batch (each image gets ANOTHER image's real pose). Tests whether the
            # CORRECT pose spatial content is causal for the LGPA gain. Test path uses true pose.
            if self.training and getattr(self, 'use_pose_shuffle', False) and scene_heatmaps is not None:
                Bp = scene_heatmaps.shape[0]
                if Bp > 1:
                    # derangement: NO image keeps its own pose (Codex: randperm leaves ~1 fixed point)
                    ar = torch.arange(Bp, device=scene_heatmaps.device)
                    perm = torch.randperm(Bp, device=ar.device)
                    tries = 0
                    while bool((perm == ar).any()) and tries < 8:
                        perm = torch.randperm(Bp, device=ar.device); tries += 1
                    if bool((perm == ar).any()):
                        perm = torch.roll(ar, 1, 0)     # guaranteed-derangement fallback (cyclic shift)
                    scene_heatmaps = scene_heatmaps[perm]
                    if target_heatmaps is not None:
                        target_heatmaps = target_heatmaps[perm]

        # Target-only heatmap swap (multi-person disambiguation).
        # Substitute scene_heatmaps with target_heatmaps so all downstream
        # pose-aware modules (PSG/LGPA/VCSR/PPA/STR/FSDC/etc.) receive the
        # target-person (index 0) signal instead of max-merged scene.
        # No other code path is touched: when use_target_heatmap is False
        # (default), scene_heatmaps keeps its original max-merged value.
        if self.use_target_heatmap and target_heatmaps is not None:
            scene_heatmaps = target_heatmaps

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

            # CLIP-ReID ID-prompt contrastive (the WORKING CLIP mechanism): align global feat
            # to per-ID learnable text prototypes via SupCon i2t/t2i.
            clip_id_loss = None
            if getattr(self, 'use_clip_id_prompt', False) and label is not None:
                from .modules.clip_id_prompt import supcon_i2t
                # Option B: per-image pose conditions the prompt (pose_vec None unless pose_cond)
                pose_vec = scene_heatmaps.float().mean(dim=(2, 3)) \
                    if (getattr(self.clip_id_prompt, 'pose_cond', False) and scene_heatmaps is not None) else None
                txt_proto = self.clip_id_prompt(label, pose_vec)  # (B, clip_dim)
                t = self.clip_id_temp
                if getattr(self, 'use_clip_id_part_guided', False) and scene_heatmaps is not None:
                    # Option C: K pose-localized part features, each aligned to the ID prototype
                    part_feats = self.pose_guided_part_pool(featmaps[-1], scene_heatmaps)  # (B, nP, C)
                    clip_id_loss = 0.0
                    for kp in range(part_feats.shape[1]):
                        ipk = self.clip_id_proj(part_feats[:, kp])
                        clip_id_loss = clip_id_loss + supcon_i2t(ipk, txt_proto, label, t) \
                            + supcon_i2t(txt_proto, ipk, label, t)
                    clip_id_loss = clip_id_loss / part_feats.shape[1]
                else:
                    # exp347 (param-free de-occluded) / Option A (pose-guided pooled) / exp341 (raw global)
                    if getattr(self, 'use_clip_id_noparam_pool', False) and scene_heatmaps is not None:
                        feat_for_clip = self.pose_weighted_pool(featmaps[-1], scene_heatmaps)
                    elif getattr(self, 'use_clip_id_pose_guided', False) and scene_heatmaps is not None:
                        feat_for_clip = self.pose_guided_pool(featmaps[-1], scene_heatmaps)
                    else:
                        feat_for_clip = global_feat
                    img_proj = self.clip_id_proj(feat_for_clip)   # (B, clip_dim)
                    clip_id_loss = supcon_i2t(img_proj, txt_proto, label, t) \
                        + supcon_i2t(txt_proto, img_proj, label, t)
                    # exp355 PGPD: pose selects a more-complete same-ID teacher; distill its
                    # soft distribution over the batch's other-ID prototypes to this student.
                    if getattr(self, 'use_pgpd', False) and scene_heatmaps is not None:
                        clip_id_loss = clip_id_loss + self._pgpd_loss(img_proj, txt_proto, label, scene_heatmaps, target_heatmaps)
                    # exp348: occluder repulsion — push the occluder-region (low-visibility) feature
                    # away from the ID prototype (penalize only positive similarity → make it neutral).
                    if getattr(self, 'use_clip_id_occ_repel', False) and scene_heatmaps is not None:
                        occ_feat = self.pose_weighted_pool(featmaps[-1], scene_heatmaps, invert=True)
                        occ_proj = torch.nn.functional.normalize(self.clip_id_proj(occ_feat), dim=1)
                        tp = torch.nn.functional.normalize(txt_proto, dim=1)
                        repel = (occ_proj * tp).sum(1).clamp(min=0).mean()
                        clip_id_loss = clip_id_loss + self.clip_id_occ_repel_w * repel

            # exp356 PC-MSC: pose-masked CLIP-semantic completion (training-only regularizer)
            if getattr(self, 'use_pcmsc', False) and scene_heatmaps is not None and self.training:
                pcmsc = self._pcmsc_loss(featmaps[-1], x, scene_heatmaps)
                clip_id_loss = pcmsc if clip_id_loss is None else clip_id_loss + pcmsc

            # VCSR: Visibility-Conditional Semantic Routing (detached)
            if getattr(self, 'use_vcsr', False) and scene_heatmaps is not None:
                vcsr_input = featmaps[-1].detach()
                vcsr_cls_scores, vcsr_feats, vcsr_data = self.vcsr_head(
                    vcsr_input, scene_heatmaps, return_cls=True)
                kp_data = vcsr_data
                return [cls_score] + vcsr_cls_scores, [global_feat] + vcsr_feats, featmaps, None, kp_data

            # Part branch: STD-PR (structural tokens) only — when GCN is NOT also enabled
            elif getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
                    and not self.use_skeleton_gcn:
                feat_map_detached = featmaps[-1].detach()
                B_fm, C_fm, H_fm, W_fm = feat_map_detached.shape
                spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)  # (B, H*W, C)
                # Pass keypoints for anchor-sampled query initialization
                kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
                sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
                router_out = self.structural_router(
                    spatial_tokens, (H_fm, W_fm), scene_heatmaps,
                    keypoints=kp_p0, scores=sc_p0,
                    input_size=tuple(x.shape[2:]))
                # Unpack: with self-attn returns (refined, stats, raw), without returns (tokens, stats)
                if getattr(self, 'str_self_attn', False):
                    structural_tokens, str_stats, raw_tokens = router_out
                else:
                    structural_tokens, str_stats = router_out
                    raw_tokens = structural_tokens
                # PLTD: Part-Level Token Dropout — randomly zero out tokens during training
                part_drop_p = getattr(self, 'str_part_drop', 0.0)
                if self.training and part_drop_p > 0:
                    B_tok, K_tok, C_tok = structural_tokens.shape
                    # Each token independently dropped with probability p
                    # Ensure at least 2 tokens survive per sample
                    drop_mask = torch.rand(B_tok, K_tok, 1, device=structural_tokens.device) >= part_drop_p
                    # Guarantee minimum 2 tokens survive
                    alive = drop_mask.squeeze(-1).sum(dim=1)  # (B,)
                    for b_idx in range(B_tok):
                        if alive[b_idx] < 2:
                            # Randomly revive tokens until we have 2
                            dead = (~drop_mask[b_idx].squeeze(-1)).nonzero(as_tuple=True)[0]
                            revive = dead[torch.randperm(len(dead))[:2 - int(alive[b_idx].item())]]
                            drop_mask[b_idx, revive] = True
                    structural_tokens = structural_tokens * drop_mask.float()
                    raw_tokens = raw_tokens * drop_mask.float()
                    n_dropped = (1 - drop_mask.float()).sum() / (B_tok * K_tok)
                    str_stats['pltd_drop'] = n_dropped.item()
                # Part feature: confidence-weighted pooling from heatmap response
                K_str = structural_tokens.shape[1]
                if K_str == 6:
                    # 6-part groups: compute per-part heatmap visibility
                    _pg = [[0,1,2,3,4],[5,6,11,12],[5,7,9],[6,8,10],[11,13,15],[12,14,16]]
                    hm_r = F.interpolate(scene_heatmaps, size=(H_fm, W_fm),
                                        mode='bilinear', align_corners=False)
                    pw = []
                    for g in _pg:
                        pw.append(hm_r[:, g].mean(dim=(1,2,3)))  # (B,)
                    part_w = torch.stack(pw, dim=1)  # (B, 6)
                    part_w = part_w / part_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    str_feat = (structural_tokens * part_w.unsqueeze(2)).sum(dim=1)
                else:
                    str_feat = structural_tokens.mean(dim=1)  # fallback
                # Per-token or pooled classification
                if getattr(self, 'str_per_token', False):
                    # DPTL dual-path: CE on raw tokens (diversity), triplet on refined tokens (coherence)
                    str_cls_list = []
                    str_feat_list = []
                    # CE path uses raw tokens (independent, diverse)
                    ce_tokens = raw_tokens
                    # Triplet/test path uses refined tokens (contextualized)
                    tri_tokens = structural_tokens
                    for k in range(ce_tokens.shape[1]):
                        tok_k = ce_tokens[:, k]  # (B, C)
                        tok_bn = self.structural_router.part_bn(tok_k)
                        str_cls_list.append(self.str_classifier(tok_bn))
                        str_feat_list.append(tri_tokens[:, k])  # refined for triplet
                    kp_data = {'str_stats': str_stats}
                    if K_str == 6:
                        kp_data['part_visibility'] = part_w  # (B, 6) per-part visibility weights
                    return [cls_score] + str_cls_list, [global_feat] + str_feat_list, featmaps, None, kp_data
                else:
                    # Pooled: all tokens averaged
                    str_feat_bn = self.structural_router.part_bn(str_feat)
                    str_cls = self.str_classifier(str_feat_bn)
                    kp_data = {'str_stats': str_stats}
                    return [cls_score, str_cls], [global_feat, str_feat], featmaps, None, kp_data

            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)):
                # LGPA: CLIP cross-attention part assignment
                lgpa_input = featmaps[-1].detach() if getattr(self, '_lgpa_detach', False) else featmaps[-1]
                lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)
                lgpa_cls_scores, lgpa_feats, lgpa_data = self.clip_part_head(
                    lgpa_input, lgpa_hm, return_cls=True)
                kp_data = lgpa_data
                if clip_id_loss is not None:
                    kp_data['clip_id_loss'] = clip_id_loss   # carry CLIP-ID-prompt loss through LGPA path

                # LGPA + GCN dual branch: also run GCN on detached features
                if self.use_skeleton_gcn and pose_dict is not None:
                    feat_map_detached = featmaps[-1].detach()
                    _s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
                    gcn_cls_scores, gcn_feats, gcn_data = self.skeleton_head(
                        feat_map_detached, pose_dict, return_cls=True, label=label,
                        stage2_feat=_s2_feat)
                    if gcn_data and 'kp_feats' in gcn_data:
                        kp_data['gcn_kp_feats'] = gcn_data['kp_feats']
                        kp_data['gcn_kp_weights'] = gcn_data['kp_weights']
                        if 'vcn_stats' in gcn_data:
                            kp_data['vcn_stats'] = gcn_data['vcn_stats']
                    return ([cls_score] + lgpa_cls_scores + gcn_cls_scores,
                            [global_feat] + lgpa_feats + gcn_feats,
                            featmaps, None, kp_data)

                return [cls_score] + lgpa_cls_scores, [global_feat] + lgpa_feats, featmaps, None, kp_data

            elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None:
                # PPA: Pose-Prompted Part-Assignment Head (end-to-end, NOT detached)
                ppa_cls_scores, ppa_feats, ppa_data = self.part_assignment_head(
                    featmaps[-1], scene_heatmaps, return_cls=True)
                kp_data = ppa_data

                # PPA + GCN dual branch: also run GCN on detached features
                if self.use_skeleton_gcn and pose_dict is not None:
                    feat_map_detached = featmaps[-1].detach()
                    _s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
                    gcn_cls_scores, gcn_feats, gcn_data = self.skeleton_head(
                        feat_map_detached, pose_dict, return_cls=True, label=label,
                        stage2_feat=_s2_feat)
                    # Merge: PPA kp_data takes priority, add GCN kp_feats for MaxSim
                    if gcn_data and 'kp_feats' in gcn_data:
                        kp_data['gcn_kp_feats'] = gcn_data['kp_feats']
                        kp_data['gcn_kp_weights'] = gcn_data['kp_weights']
                    return ([cls_score] + ppa_cls_scores + gcn_cls_scores,
                            [global_feat] + ppa_feats + gcn_feats,
                            featmaps, None, kp_data)

                return [cls_score] + ppa_cls_scores, [global_feat] + ppa_feats, featmaps, None, kp_data

            elif self.use_skeleton_gcn and pose_dict is not None:
                # GSPB: Gradient-scaled part branch
                # scale=0 → detach (default), scale=1 → non-detach
                _gs = getattr(self, '_part_grad_scale', 0.0)
                if _gs > 0:
                    feat_map_detached = featmaps[-1].detach() + _gs * (featmaps[-1] - featmaps[-1].detach())
                else:
                    feat_map_detached = featmaps[-1].detach()

                # Dual Part Branch: also run STD-PR for per-token SupCon if both are enabled
                dual_branch_active = False
                if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
                        and getattr(self, 'str_per_token', False):
                    B_fm, C_fm, H_fm, W_fm = feat_map_detached.shape
                    spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)
                    kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
                    sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
                    router_out = self.structural_router(
                        spatial_tokens, (H_fm, W_fm), scene_heatmaps,
                        keypoints=kp_p0, scores=sc_p0,
                        input_size=tuple(x.shape[2:]))
                    if getattr(self, 'str_self_attn', False):
                        structural_tokens, str_stats, raw_tokens = router_out
                    else:
                        structural_tokens, str_stats = router_out
                        raw_tokens = structural_tokens
                    # Per-token classification for SupCon
                    ce_tokens = raw_tokens
                    tri_tokens = structural_tokens
                    str_cls_list = []
                    str_feat_list = []
                    for k in range(ce_tokens.shape[1]):
                        tok_k = ce_tokens[:, k]
                        tok_bn = self.structural_router.part_bn(tok_k)
                        str_cls_list.append(self.str_classifier(tok_bn))
                        str_feat_list.append(tri_tokens[:, k])
                    dual_branch_active = True

                # FSDC: Feature-Space Diffusion Completion
                fsdc_loss = None
                if getattr(self, 'use_fsdc', False):
                    B_d, C_d, H_d, W_d = feat_map_detached.shape
                    spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)  # (B, N, C)
                    completed_tokens, fsdc_loss, fsdc_stats = self.feature_denoiser(
                        spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
                    # Reshape back to feature map
                    feat_map_detached = completed_tokens.transpose(1, 2).reshape(B_d, C_d, H_d, W_d)

                # Pass Stage 2 features for KAMP/MRKF multi-scale fusion
                _s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
                gcn_cls_scores, gcn_feats, kp_data = self.skeleton_head(
                    feat_map_detached, pose_dict, return_cls=True, label=label,
                    stage2_feat=_s2_feat)
                # Store FSDC loss in kp_data for processor
                if fsdc_loss is not None:
                    if kp_data is None:
                        kp_data = {}
                    kp_data['fsdc_loss'] = fsdc_loss
                    kp_data['fsdc_stats'] = fsdc_stats

                # PNIS: normalize GCN feature by subtracting pose offset
                if getattr(self, 'use_pose_normalize', False) and len(gcn_feats) > 0:
                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2) person 0
                    kp_scores = pose_dict['scores'][:, 0, :]        # (B, 17) person 0
                    # Normalize coordinates to [0,1]
                    img_h, img_w = x.shape[2], x.shape[3]
                    kp_coords_norm = kp_coords.clone()
                    kp_coords_norm[:, :, 0] = kp_coords_norm[:, :, 0] / max(img_w, 1)
                    kp_coords_norm[:, :, 1] = kp_coords_norm[:, :, 1] / max(img_h, 1)
                    identity_feat, pn_stats = self.pose_normalizer(
                        gcn_feats[0], kp_coords_norm, kp_scores)
                    gcn_feats[0] = identity_feat
                    if kp_data is None:
                        kp_data = {}
                    kp_data['pn_stats'] = pn_stats

                # SPLADE: auxiliary sparse classification (does NOT modify gcn lists)
                if getattr(self, 'use_splade', False) and len(gcn_feats) > 0:
                    sparse_feat, sparsity = self.sparse_head(gcn_feats[0])
                    sparse_cls = self.sparse_classifier(sparse_feat)
                    if kp_data is None:
                        kp_data = {}
                    kp_data['splade_cls'] = sparse_cls      # separate CE loss in processor
                    kp_data['splade_sparsity'] = sparsity
                    kp_data['splade_reg'] = sparse_feat.mean()  # sparsity regularization

                # Dual Part Branch: combine GCN + STD-PR per-token outputs
                if dual_branch_active:
                    # Return: [global, str_tok1..6, gcn] for both scores and feats
                    # SupCon operates on str_tok1..6, GCN provides architecture via gcn
                    if kp_data is None:
                        kp_data = {}
                    kp_data['str_stats'] = str_stats
                    kp_data['num_str_tokens'] = len(str_feat_list)  # SupCon uses feat[1:1+num_str_tokens]
                    # part_visibility for STD-PR tokens
                    K_str = len(str_feat_list)
                    if K_str == 6:
                        _pg = [[0,1,2,3,4],[5,6,11,12],[5,7,9],[6,8,10],[11,13,15],[12,14,16]]
                        hm_r = F.interpolate(scene_heatmaps, size=(featmaps[-1].shape[2], featmaps[-1].shape[3]),
                                            mode='bilinear', align_corners=False)
                        pw = [hm_r[:, g].mean(dim=(1,2,3)) for g in _pg]
                        part_w = torch.stack(pw, dim=1)
                        part_w = part_w / part_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
                        kp_data['part_visibility'] = part_w
                    return ([cls_score] + str_cls_list + gcn_cls_scores,
                            [global_feat] + str_feat_list + gcn_feats,
                            featmaps, None, kp_data)

                # BA-PKC: sample keypoint features from NON-detached feature map
                # Gradients flow to backbone, unlike GCN which uses detached features
                if getattr(self, 'ba_pkc', False) or getattr(self, 'bt_pkd', False):
                    raw_fm = featmaps[-1]  # (B, C, fH, fW) — NOT detached!
                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
                    input_h, input_w = x.shape[2], x.shape[3]
                    grid_x = (kp_coords[:, :, 0] / input_w * 2 - 1).clamp(-1, 1)
                    grid_y = (kp_coords[:, :, 1] / input_h * 2 - 1).clamp(-1, 1)
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, 17, 1, 2)
                    sampled = F.grid_sample(raw_fm, grid, mode='bilinear',
                                            padding_mode='border', align_corners=True)
                    ba_kp_feats = sampled.squeeze(-1).permute(0, 2, 1)  # (B, 17, C)
                    if kp_data is None:
                        kp_data = {}
                    if getattr(self, 'ba_pkc', False):
                        kp_data['ba_kp_feats'] = ba_kp_feats
                    if getattr(self, 'bt_pkd', False):
                        kp_data['bt_kp_feats'] = ba_kp_feats  # non-detached for distillation

                # Return lists -> triggers list-loss path (implicit 0.5x global)
                return [cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, featmaps, None, kp_data

            if clip_id_loss is not None:
                return cls_score, global_feat, featmaps, None, {'clip_id_loss': clip_id_loss}
            return cls_score, global_feat, featmaps, None
        else:
            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            # Part branch test features
            gcn_feats = None
            aux_data = {}

            # VCSR test path
            if getattr(self, 'use_vcsr', False) and scene_heatmaps is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                _, vcsr_feats, aux_data = self.vcsr_head(
                    featmaps[-1], scene_heatmaps, return_cls=False)
                gcn_feats = vcsr_feats

            # LGPA test path — uses scene_heatmaps (same as PPA for fair comparison)
            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)) and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)
                _, lgpa_feats, aux_data = self.clip_part_head(
                    featmaps[-1], lgpa_hm, return_cls=False)
                gcn_feats = lgpa_feats  # [pooled, part1..partK]
                # LGPA + GCN dual: also get GCN features
                if self.use_skeleton_gcn and pose_dict is not None:
                    _, gcn_only_feats, gcn_aux = self.skeleton_head(
                        featmaps[-1], pose_dict, return_cls=False)
                    gcn_feats = lgpa_feats + gcn_only_feats
                    if gcn_aux and 'kp_feats' in gcn_aux:
                        aux_data['gcn_kp_feats'] = gcn_aux['kp_feats']
                        aux_data['gcn_kp_weights'] = gcn_aux['kp_weights']

            # PPA test path
            elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                _, ppa_feats, aux_data = self.part_assignment_head(
                    featmaps[-1], scene_heatmaps, return_cls=False)
                gcn_feats = ppa_feats  # [pooled, part1..partK]
                # PPA + GCN dual: also get GCN features
                if self.use_skeleton_gcn and pose_dict is not None:
                    _, gcn_only_feats, gcn_aux = self.skeleton_head(
                        featmaps[-1], pose_dict, return_cls=False)
                    gcn_feats = ppa_feats + gcn_only_feats
                    if gcn_aux and 'kp_feats' in gcn_aux:
                        aux_data['gcn_kp_feats'] = gcn_aux['kp_feats']
                        aux_data['gcn_kp_weights'] = gcn_aux['kp_weights']

            elif getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global' and not self.use_skeleton_gcn:
                B_fm, C_fm, H_fm, W_fm = featmaps[-1].shape
                spatial_tokens = featmaps[-1].flatten(2).transpose(1, 2)
                kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
                sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
                router_out = self.structural_router(
                    spatial_tokens, (H_fm, W_fm), scene_heatmaps,
                    keypoints=kp_p0, scores=sc_p0,
                    input_size=tuple(x.shape[2:]))
                # Use refined tokens (first return) for test features
                structural_tokens = router_out[0]
                # Confidence-weighted pooling (same as training)
                K_str = structural_tokens.shape[1]
                if K_str == 6:
                    _pg = [[0,1,2,3,4],[5,6,11,12],[5,7,9],[6,8,10],[11,13,15],[12,14,16]]
                    hm_r = F.interpolate(scene_heatmaps, size=(H_fm, W_fm),
                                        mode='bilinear', align_corners=False)
                    pw = [hm_r[:, g].mean(dim=(1,2,3)) for g in _pg]
                    part_w = torch.stack(pw, dim=1)
                    part_w = part_w / part_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    str_feat = (structural_tokens * part_w.unsqueeze(2)).sum(dim=1)
                else:
                    str_feat = structural_tokens.mean(dim=1)
                if self.pose_test_feat in ('maxsim', 'maxsim_hybrid',
                                          'cvk_hybrid', 'cvk_only'):
                    # Return structural tokens as kp_feats for set matching
                    K = structural_tokens.shape[1]
                    test_feat = {
                        'mode': self.pose_test_feat,
                        'global_feat': test_feat,
                        'kp_feats': structural_tokens,  # (B, K, C)
                        'kp_weights': torch.ones(structural_tokens.shape[0], K,
                                                 device=structural_tokens.device),
                    }
                    return test_feat, featmaps
                # Per-token training uses pooled test feature (better than per-token concat)
                # Confidence-weighted pool captures the right signal; per-token concat dilutes it
                gcn_feats = [str_feat]  # equal_concat: global + pooled_part
            elif self.use_skeleton_gcn and pose_dict is not None and \
                    getattr(self, 'pose_test_feat', 'global') != 'global':
                # FSDC: complete occluded tokens at test time
                feat_for_gcn = featmaps[-1]
                if getattr(self, 'use_fsdc', False) and scene_heatmaps is not None:
                    B_d, C_d, H_d, W_d = feat_for_gcn.shape
                    spatial_tokens = feat_for_gcn.flatten(2).transpose(1, 2)
                    completed, _, _ = self.feature_denoiser(
                        spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
                    feat_for_gcn = completed.transpose(1, 2).reshape(B_d, C_d, H_d, W_d)
                _s2_test = featmaps[-2] if len(featmaps) >= 2 else None
                _, gcn_feats, aux_data = self.skeleton_head(
                    feat_for_gcn, pose_dict, return_cls=False,
                    stage2_feat=_s2_test)
                # Dual Part Branch test: also add STD-PR per-token features
                if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
                        and getattr(self, 'str_per_token', False):
                    B_fm, C_fm, H_fm, W_fm = featmaps[-1].shape
                    spatial_tokens = featmaps[-1].flatten(2).transpose(1, 2)
                    kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
                    sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
                    router_out = self.structural_router(
                        spatial_tokens, (H_fm, W_fm), scene_heatmaps,
                        keypoints=kp_p0, scores=sc_p0,
                        input_size=tuple(x.shape[2:]))
                    structural_tokens = router_out[0]
                    # Add each structural token to gcn_feats for equal_concat
                    for k in range(structural_tokens.shape[1]):
                        gcn_feats.append(structural_tokens[:, k])
                # PNIS: normalize test features too
                if getattr(self, 'use_pose_normalize', False) and gcn_feats is not None and len(gcn_feats) > 0:
                    kp_coords = pose_dict['keypoints'][:, 0, :, :]
                    kp_scores = pose_dict['scores'][:, 0, :]
                    img_h, img_w = x.shape[2], x.shape[3]
                    kp_coords_norm = kp_coords.clone()
                    kp_coords_norm[:, :, 0] /= max(img_w, 1)
                    kp_coords_norm[:, :, 1] /= max(img_h, 1)
                    identity_feat, _ = self.pose_normalizer(gcn_feats[0], kp_coords_norm, kp_scores)
                    gcn_feats[0] = identity_feat
                # SPLADE: training-only auxiliary, no test-time feature change

            # Assemble test features from global + part branch
            if gcn_feats is not None:
                if self.pose_test_feat == 'gcn_only':
                    test_feat = torch.cat(gcn_feats, dim=1)
                elif self.pose_test_feat == 'equal_concat':
                    g_norm = F.normalize(test_feat, p=2, dim=1)
                    p_norm = [F.normalize(f, p=2, dim=1) for f in gcn_feats]
                    test_feat = torch.cat([g_norm] + p_norm, dim=1)
                elif self.pose_test_feat == 'part_only':
                    # Diagnostic: LGPA/part branch vectors ONLY (drop global), each L2-normed
                    test_feat = torch.cat(
                        [F.normalize(f, p=2, dim=1) for f in gcn_feats], dim=1)
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
