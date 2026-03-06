import logging
import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = True if pretrain_choice == 'imagenet' else False

        # Build extra kwargs for VPReID or with_cp support
        extra_kwargs = {}
        with_cp = getattr(cfg.MODEL, 'WITH_CP', False)
        if with_cp:
            extra_kwargs['with_cp'] = True

        vpreid_cfg = getattr(cfg.MODEL, 'VPREID', None)
        if vpreid_cfg is not None and getattr(vpreid_cfg, 'ENABLE', False):
            extra_kwargs['pose_cfg'] = vpreid_cfg.POSE_CFG
            extra_kwargs['pose_ckpt'] = vpreid_cfg.POSE_CKPT
            extra_kwargs['n_parts'] = vpreid_cfg.N_PARTS
            extra_kwargs['part_temp'] = vpreid_cfg.PART_TEMP
            extra_kwargs['vis_threshold'] = vpreid_cfg.VIS_THRESHOLD

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrained=model_path,
            convert_weights=convert_weights,
            semantic_weight=semantic_weight,
            **extra_kwargs,
        )
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        # Detect VPReID backbone
        self.is_vpreid = getattr(self.base, 'is_vpreid', False)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        if self.is_vpreid:
            K = self.base.n_body_parts
            D = self.in_planes
            # Global BN + classifier
            self.global_bnneck = nn.BatchNorm1d(D)
            self.global_bnneck.bias.requires_grad_(False)
            self.global_bnneck.apply(weights_init_kaiming)
            self.classifier_global = nn.Linear(D, self.num_classes, bias=False)
            self.classifier_global.apply(weights_init_classifier)
            # Foreground BN + classifier
            self.fg_bnneck = nn.BatchNorm1d(D)
            self.fg_bnneck.bias.requires_grad_(False)
            self.fg_bnneck.apply(weights_init_kaiming)
            self.classifier_fg = nn.Linear(D, self.num_classes, bias=False)
            self.classifier_fg.apply(weights_init_classifier)
            # Per-part BN + classifiers
            self.part_bnnecks = nn.ModuleList()
            self.part_classifiers = nn.ModuleList()
            for _ in range(K):
                bn = nn.BatchNorm1d(D)
                bn.bias.requires_grad_(False)
                bn.apply(weights_init_kaiming)
                self.part_bnnecks.append(bn)
                cls = nn.Linear(D, self.num_classes, bias=False)
                cls.apply(weights_init_classifier)
                self.part_classifiers.append(cls)
            self.dropout = nn.Dropout(self.dropout_rate)

        elif self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        # --- PosePart: offline pose-guided part features ---
        self.use_pose_part = False
        pose_part_cfg = getattr(cfg.MODEL, 'POSE_PART', None)
        if pose_part_cfg is not None and getattr(pose_part_cfg, 'ENABLE', False):
            from .modules.pose_part import PosePartPooling, PosePartHead
            self.use_pose_part = True
            self.pose_part_pool = PosePartPooling(
                n_parts=pose_part_cfg.N_PARTS,
                sigma=pose_part_cfg.SIGMA,
                img_size=cfg.INPUT.SIZE_TRAIN,
            )
            self.pose_part_head = PosePartHead(
                in_channels=self.in_planes,
                num_classes=num_classes,
                n_parts=pose_part_cfg.N_PARTS,
            )
            print(f'===========PosePart enabled: {pose_part_cfg.N_PARTS} parts, sigma={pose_part_cfg.SIGMA}===========')

        # --- PCFC: Pose-Conditioned Feature Calibration ---
        self.use_pcfc = False
        pcfc_cfg = getattr(cfg.MODEL, 'PCFC', None)
        if pcfc_cfg is not None and getattr(pcfc_cfg, 'ENABLE', False):
            from .modules.pose_calibration import PoseFeatureCalibration
            from .modules.pose_part import PosePartHead
            self.use_pcfc = True
            self.pcfc = PoseFeatureCalibration(
                img_size=cfg.INPUT.SIZE_TRAIN,
                sigma=pcfc_cfg.SIGMA,
                alpha_init=pcfc_cfg.ALPHA_INIT,
                use_part_loss=pcfc_cfg.USE_PART_LOSS,
                n_parts=pcfc_cfg.N_PARTS,
                part_sigma=pcfc_cfg.PART_SIGMA,
            )
            self.pcfc_gap = nn.AdaptiveAvgPool2d(1)
            if pcfc_cfg.USE_PART_LOSS:
                self.pcfc_part_head = PosePartHead(
                    in_channels=self.in_planes,
                    num_classes=num_classes,
                    n_parts=pcfc_cfg.N_PARTS,
                )
            print(f'===========PCFC enabled: sigma={pcfc_cfg.SIGMA}, alpha_init={pcfc_cfg.ALPHA_INIT}===========')

        # --- PVFM: Pose-Guided Visibility Feature Modulation ---
        self.use_pvfm = False
        pvfm_cfg = getattr(cfg.MODEL, 'PVFM', None)
        if pvfm_cfg is not None and getattr(pvfm_cfg, 'ENABLE', False):
            from .modules.pose_vis_modulation import PoseVisFeatureModulation
            self.use_pvfm = True
            active_stages = tuple(pvfm_cfg.ACTIVE_STAGES) if hasattr(pvfm_cfg, 'ACTIVE_STAGES') else (2, 3)
            self.pvfm = PoseVisFeatureModulation(
                n_stages=4,
                img_size=cfg.INPUT.SIZE_TRAIN,
                sigma=pvfm_cfg.SIGMA,
                beta_init=pvfm_cfg.BETA_INIT,
                active_stages=active_stages,
            )
            print(f'===========PVFM enabled: stages={active_stages}, sigma={pvfm_cfg.SIGMA}, beta_init={pvfm_cfg.BETA_INIT}===========')

        # --- KPE: Keypoint Prompt Embedding ---
        self.use_kpe = False
        kpe_cfg = getattr(cfg.MODEL, 'KPE', None)
        if kpe_cfg is not None and getattr(kpe_cfg, 'ENABLE', False):
            from .modules.keypoint_prompt import KeypointPromptEmbedding
            self.use_kpe = True
            # Initial embed dim is num_features[0] (96 for Swin-Tiny)
            initial_embed_dim = self.base.num_features[0]
            self.kpe = KeypointPromptEmbedding(
                embed_dim=initial_embed_dim,
                img_size=cfg.INPUT.SIZE_TRAIN,
                patch_size=4,  # Swin-Tiny default
                sigma=kpe_cfg.SIGMA,
            )
            print(f'===========KPE enabled: embed_dim={initial_embed_dim}, sigma={kpe_cfg.SIGMA}===========')

        #if pretrain_choice == 'self':
        #    self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None,
                keypoints=None, visibility=None):
        # --- VPReID path ---
        if self.is_vpreid:
            outputs = self.base(x)
            g_feat = outputs['global_feat']
            fg_feat = outputs['foreground_feat']
            part_feats = outputs['part_feats']

            if self.training:
                g_bn = self.global_bnneck(g_feat)
                fg_bn = self.fg_bnneck(fg_feat)
                scores = [self.classifier_global(g_bn), self.classifier_fg(fg_bn)]

                K = part_feats.shape[1]
                feat_list = [g_feat, fg_feat]
                for k in range(K):
                    pk_bn = self.part_bnnecks[k](part_feats[:, k])
                    feat_list.append(pk_bn)
                    scores.append(self.part_classifiers[k](pk_bn))

                extras = {'part_vis': outputs['part_vis']}
                return scores, feat_list, extras
            else:
                g_bn = self.global_bnneck(g_feat)
                # Return global feature compatible with standard eval path
                if self.neck_feat == 'after':
                    return g_bn, None
                else:
                    return g_feat, None

        # --- Standard path ---
        # Build extra kwargs for backbone
        backbone_kwargs = {}
        if self.use_pvfm and keypoints is not None:
            backbone_kwargs.update(dict(
                vis_modulation=self.pvfm,
                keypoints=keypoints,
                visibility=visibility,
            ))
        if self.use_kpe and keypoints is not None:
            backbone_kwargs['kpe_module'] = self.kpe
            backbone_kwargs['keypoints'] = keypoints
            backbone_kwargs['visibility'] = visibility
        global_feat, featmaps = self.base(x, **backbone_kwargs)

        # --- PCFC: re-pool with visibility attention ---
        if self.use_pcfc and keypoints is not None and featmaps is not None:
            last_fm = featmaps[-1] if isinstance(featmaps, (list, tuple)) else featmaps
            calibrated_fm, attn_map, part_feats, part_vis = self.pcfc(
                last_fm, keypoints, visibility
            )
            # Re-pool the calibrated feature map → occlusion-aware global feature
            global_feat = self.pcfc_gap(calibrated_fm).flatten(1)  # [B, C]

        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        # --- PCFC branch (training / eval) ---
        if self.use_pcfc and keypoints is not None and featmaps is not None:
            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat_cls, label)
                else:
                    cls_score = self.classifier(feat_cls)

                extras = {'attn_alpha': self.pcfc.vis_attn.alpha.item()}
                # Add KPE scale if active
                if self.use_kpe:
                    extras['kpe_scale'] = self.kpe.scale.item()
                # Add PVFM beta values if active
                if self.use_pvfm:
                    for s, mod in self.pvfm.stage_mods.items():
                        extras[f'beta_s{s}'] = mod.beta.item()
                if part_feats is not None:
                    part_logits, part_feats_bn = self.pcfc_part_head(part_feats, part_vis)
                    extras['part_logits'] = part_logits
                    extras['part_vis'] = part_vis
                    extras['part_feats'] = part_feats_bn  # [B, K, C] for part triplet
                return cls_score, global_feat, extras
            else:
                if self.neck_feat == 'after':
                    return feat, None
                else:
                    return global_feat, None

        # --- PosePart branch ---
        if self.use_pose_part and keypoints is not None and featmaps is not None:
            part_feats, part_vis = self.pose_part_pool(featmaps, keypoints, visibility)

            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat_cls, label)
                else:
                    cls_score = self.classifier(feat_cls)

                part_logits, part_feats_bn = self.pose_part_head(part_feats, part_vis)
                extras = {
                    'part_logits': part_logits,
                    'part_vis': part_vis,
                }
                return cls_score, global_feat, extras
            else:
                # Test: concatenate global + visibility-weighted part features
                if self.neck_feat == 'after':
                    part_cat = self.pose_part_head(part_feats, part_vis)  # [B, K*C]
                    return torch.cat([feat, part_cat], dim=1), None
                else:
                    part_cat = self.pose_part_head(part_feats, part_vis)
                    return torch.cat([global_feat, part_cat], dim=1), None

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat, featmaps  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return feat, featmaps
            else:
                return global_feat, featmaps

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model


# --- register VPReID types ---
from .backbones.vpreid import (
    vpreid_tiny_patch4_window7_224,
    vpreid_small_patch4_window7_224,
    vpreid_base_patch4_window7_224,
)
__factory_T_type.update({
    'vpreid_tiny_patch4_window7_224': vpreid_tiny_patch4_window7_224,
    'vpreid_small_patch4_window7_224': vpreid_small_patch4_window7_224,
    'vpreid_base_patch4_window7_224': vpreid_base_patch4_window7_224,
})
