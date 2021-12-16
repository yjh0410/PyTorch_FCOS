import torch
from .fcos import FCOS
from .fcos_rt import FCOS_RT


# build FCOS detector
def build_model(args, cfg, device, num_classes=80, trainable=False):
    if args.model == 'fcos':
        print('Build FCOS ...')
        model = FCOS(cfg=cfg,
                        device=device,
                        num_classes=num_classes,
                        trainable=trainable,
                        norm=args.norm,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh)
    elif args.model == 'fcos_rt':
        print('Build FCOS-RT ...')
        model = FCOS_RT(cfg=cfg,
                        device=device,
                        num_classes=num_classes,
                        trainable=trainable,
                        norm=args.norm,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh)

    return model
