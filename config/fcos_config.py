# FCOS config


fcos_config = {
    # 1x
    'fcos_r50_fpn_1x': {
        # model
        'backbone': 'resnet50',
        'head_dims': 256,
        'fpn': 'basic_fpn',
        'norm': 'GN',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # epoch
        'max_epoch': 12,
        'lr_epoch': [8, 11],
        # matcher
        'scale_range': [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1e10)],
    },
    # 1x
    'fcos_r101_fpn_1x': {
        # model
        'backbone': 'resnet101',
        'head_dims': 256,
        'fpn': 'basic_fpn',
        'norm': 'GN',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # epoch
        'max_epoch': 12,
        'lr_epoch': [8, 11],
        # matcher
        'scale_range': [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1e10)],
    },
    # 4x
    'fcos_rt_r50_fpn_4x': {
        # model
        'backbone': 'resnet50',
        'head_dims': 256,
        'fpn': 'basic_fpn',
        'norm': 'GN',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # epoch
        'max_epoch': 48,
        'lr_epoch': [16, 22],
        # matcher
        'scale_range': [(0, 64), (64, 128), (128, 1e10)],
    }
}