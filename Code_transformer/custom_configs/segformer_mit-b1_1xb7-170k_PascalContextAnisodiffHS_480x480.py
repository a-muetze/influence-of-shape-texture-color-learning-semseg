output_dir = 'zzz_myresults'
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        79.664907 ,
        113.387535
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        480,
        480,
    ),
    std=[
      49.7458335,
      65.84151 
    ],
    type='SegDataPreProcessor')
data_root = '/home/'
dataset_type = 'CustomPascalContextDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1700, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=2,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=33,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
# optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1700, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1700,
        by_epoch=False,
        end=170000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(out_channels=2, type='RGB2HS'),
    dict(type='LoadAnnotations'),
    dict(type='CropValPC33'),
    dict(type='PackSegInputs'),
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='JPEGImages/test',
            seg_map_path='SemsegIDImg/test'
            ),
        data_root='/home/xxx/data/Pascalcontext/',
        pipeline=
            test_pipeline,
        type='CustomPascalContextDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    output_dir=output_dir,
    type='IoUMetric',
    prefix='test')

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(out_channels=2, type='RGB2HS'),
    dict(type='LoadAnnotations'),
    dict(type='CropTrainPC33'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=170000, type='IterBasedTrainLoop', val_interval=1700)
train_dataloader = dict(
    batch_size=7,
    dataset=dict(
        data_prefix=dict(
            img_path='EEDImages/train',
            seg_map_path='SemsegIDImg/train'
            ),
        data_root='/home/xxx/data/Pascalcontext/',
        pipeline=
            train_pipeline,
        type='CustomPascalContextDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='EEDImages/val',
            seg_map_path='SemsegIDImg/val'),
        data_root='/home/xxx/data/Pascalcontext/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='unchanged'),
            dict(out_channels=2, type='RGB2HS'),
            dict(type='LoadAnnotations'),
            dict(type='CropValPC33'),
            dict(type='PackSegInputs'),
        ],
        type='CustomPascalContextDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    prefix='val',
    type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
    