"""Minimal inference config for ViTPose + visibility head.

This matches the custom checkpoint `pretrained/best_coco_AP_epoch_210.pth`,
which wraps a standard HeatmapHead with MMPose's `VisPredictHead`.
The file is intentionally self-contained so it can be used on the current
branch without relying on config inheritance from other branches.
"""

default_scope = 'mmpose'

codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=None,
    ),
    head=dict(
        type='VisPredictHead',
        loss=dict(
            type='BCELoss',
            use_target_weight=True,
            use_sigmoid=True,
            loss_weight=1e-3,
        ),
        pose_cfg=dict(
            type='HeatmapHead',
            in_channels=768,
            out_channels=17,
            deconv_out_channels=(256, 256),
            deconv_kernel_sizes=(4, 4),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=codec,
        ),
    ),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
    ),
)

test_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
            dict(type='PackPoseInputs'),
        ],
    ),
)
