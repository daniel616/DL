# model settings
input_size = 512
model = dict(
    type='SingleStageDetector',
    pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='SSDVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=2,
        anchor_strides=(8, 16, 32, 64, 100, 300),
        basesize_ratio_range=(0.15, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings

# dataset settings
dataset_type =  'DL_coco'
data_root = 'data/deeplesion/'
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'DL_train_toy.csv',
        img_prefix=data_root + 'Images_png/',
        img_scale=(512, 512),
        size_divisor=32,
        flip_ratio=0.5,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'DL_val_toy.csv',
        img_prefix=data_root + 'Images_png/',
        img_scale=(512, 512),
        size_divisor=32,
        flip_ratio=0,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'DL_test_toy.csv',
        img_prefix=data_root + 'Images_png/',
        img_scale=(512,512),
        size_divisor=32,
        flip_ratio=0,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssd300_coco'
load_from = None
resume_from = None
workflow = [('train', 1)]
