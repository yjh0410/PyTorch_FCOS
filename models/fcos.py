import numpy as np
import torch
import torch.nn as nn
from .conv import Conv
from .resnet import build_backbone
from .fpn import build_fpn


class FCOS(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 norm = 'BN'):
        super(FCOS, self).__init__()
        self.device = device
        self.fmp_size = None
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32, 64, 128]

        # backbone
        self.backbone, feature_channels = build_backbone(model_name=cfg['backbone'],
                                                         pretrained=trainable,
                                                         train_backbone=True,
                                                         return_interm_layers=True)

        # neck
        self.neck = build_fpn(model_name=cfg['fpn'], in_channels=feature_channels, out_channel=cfg['head_dims'])

        # head
        self.cls_feat = nn.Sequential(
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm)
        )
        self.reg_feat = nn.Sequential(
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=norm)
        )

        # head
        self.cls_pred = nn.Conv2d(cfg['head_dims'], self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(cfg['head_dims'], 4, kernel_size=1)
        self.ctn_pred = nn.Conv2d(cfg['head_dims'], 1, kernel_size=1)

        if self.trainable:
            # init bias
            self._init_head()


    def _init_head(self):  
        # init weight of decoder
        for m in [self.cls_feat, self.reg_feat]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
                
        # init bias of cls_head
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @torch.no_grad()
    def inference_single_image(self, x):
        """
            image: (tensor) [1, 3, H, W]
        """
        img_h, img_w = x.shape[-2:]
        # backbone: C3, C4, C5
        x = self.backbone(x)

        # neck: P3, P4, P5
        features = self.neck(x)

        outputs = {
            "pred_cls": [],
            "pred_box": [],
            "pred_ctn": []
        }
        # head
        for i, p in enumerate(features):
            fmp_h_i, fmp_w_i = p.shape[-2:]
            cls_feat_i = self.cls_feat(p)
            reg_feat_i = self.reg_feat(p)
            # pred
            cls_pred_i = self.cls_pred(cls_feat_i)
            reg_pred_i = self.reg_pred(reg_feat_i)
            ctn_pred_i = self.ctn_pred(reg_feat_i)

            # [1, C, H, W] -> [H, W, C]
            cls_pred_i = cls_pred_i.permute(0, 2, 3, 1).contiguous()[0]
            reg_pred_i = reg_pred_i.permute(0, 2, 3, 1).contiguous()[0]
            ctn_pred_i = ctn_pred_i.permute(0, 2, 3, 1).contiguous()[0]
        
            # decode box
            ## generate grid cells
            anchor_y_i, anchor_x_i = torch.meshgrid([torch.arange(fmp_h_i), torch.arange(fmp_w_i)])
            # [H, W, 2]
            anchor_xy_i = torch.stack([anchor_x_i, anchor_y_i], dim=-1).float() + 0.5
            # [H, W, 2] -> [H, W, 2]
            anchor_xy_i = anchor_xy_i.to(self.device)

            ## decode box
            x1y1_pred_i = anchor_xy_i - reg_pred_i[..., :2].relu()
            x2y2_pred_i = anchor_xy_i + reg_pred_i[..., 2:].relu()
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)
            # rescale
            box_pred_i = box_pred_i * self.stride[i]

            # [H, W, C] -> [HW, C]
            outputs["pred_cls"].append(cls_pred_i.view(-1, self.num_classes))
            outputs["pred_box"].append(box_pred_i.view(-1, 4))
            outputs["pred_ctn"].append(ctn_pred_i.view(-1, 1))

        outputs["pred_cls"] = torch.cat(outputs["pred_cls"], dim=0)  # [HW, C]
        outputs["pred_box"] = torch.cat(outputs["pred_box"], dim=0)  # [HW, 4]
        outputs["pred_ctn"] = torch.cat(outputs["pred_ctn"], dim=0)  # [HW, 1]

        # score = sqrt(cls * ctn)
        scores = torch.sqrt(outputs["pred_cls"].sigmoid() * \
                            outputs["pred_ctn"].sigmoid()).view(-1, self.num_classes) # [HW, C]

        # normalize bbox
        bboxes = outputs["pred_box"].view(-1, 4) # [HW, 4]
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clamp(0., 1.)

        # to cpu
        scores = scores.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, image, masks=None):
        """
            image: (tensor) [B, 3, H, W]
            mask: (tensor) [B, H, W]
        """
        if not self.trainable:
            # inference
            bboxes, scores, cls_inds = self.inference_single_image(image)
            return bboxes, scores, cls_inds

        else:
            B = image.size(0)
            # backbone: C3, C4, C5
            x = self.backbone(image)

            # neck: P3, P4, P5, P6, P7
            features = self.neck(x)

            outputs = {
                "pred_cls": [],
                "pred_box": [],
                "pred_ctn": [],
                "masks": [],
                "fmp_size": [],
                "strides": []
            }
            # head
            print(len(features))
            for i, p in enumerate(features):
                fmp_h_i, fmp_w_i = p.shape[-2:]
                cls_feat_i = self.cls_feat(p)
                reg_feat_i = self.reg_feat(p)
                # pred
                cls_pred_i = self.cls_pred(cls_feat_i)
                reg_pred_i = self.reg_pred(reg_feat_i)
                ctn_pred_i = self.ctn_pred(reg_feat_i)

                # [B, C, H, W] -> [B, H, W, C]
                cls_pred_i = cls_pred_i.permute(0, 2, 3, 1).contiguous()
                reg_pred_i = reg_pred_i.permute(0, 2, 3, 1).contiguous()
                ctn_pred_i = ctn_pred_i.permute(0, 2, 3, 1).contiguous()
            
                # decode box
                ## generate grid cells
                anchor_y_i, anchor_x_i = torch.meshgrid([torch.arange(fmp_h_i), torch.arange(fmp_w_i)])
                # [H, W, 2]
                anchor_xy_i = torch.stack([anchor_x_i, anchor_y_i], dim=-1).float() + 0.5
                # [H, W, 2] -> [1, H, W, 2]
                anchor_xy_i = anchor_xy_i.unsqueeze(0).to(self.device)

                ## decode box
                x1y1_pred_i = anchor_xy_i - reg_pred_i[..., :2].exp()
                x2y2_pred_i = anchor_xy_i + reg_pred_i[..., 2:].exp()
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)

                outputs["pred_cls"].append(cls_pred_i.view(B, -1, self.num_classes))
                outputs["pred_box"].append(box_pred_i.view(B, -1, 4))
                outputs["pred_ctn"].append(ctn_pred_i.view(B, -1, 1))
                outputs["fmp_size"].append([fmp_h_i, fmp_w_i])
                outputs["strides"].append(self.stride[i])

                # mask
                if masks is not None:
                    # [B, H, W]
                    resized_masks = torch.nn.functional.interpolate(masks[None], size=[fmp_h_i, fmp_w_i]).bool()[0]
                    # [B, HW]
                    resized_masks = resized_masks.flatten(1)
                    outputs["masks"].append(resized_masks)

            outputs["pred_cls"] = torch.cat(outputs["pred_cls"], dim=1)  # [B, HW, C]
            outputs["pred_box"] = torch.cat(outputs["pred_box"], dim=1)  # [B, HW, 4]
            outputs["pred_ctn"] = torch.cat(outputs["pred_ctn"], dim=1)  # [B, HW, 1]
            outputs["masks"] = torch.cat(outputs["masks"], dim=1) if outputs["masks"] else []      # [B, HW, 1]

            return outputs


# build FCOS detector
def build_model(args, cfg, device, num_classes=80, trainable=False):
    model = FCOS(cfg=cfg,
                    device=device,
                    num_classes=num_classes,
                    trainable=trainable,
                    norm=args.norm,
                    conf_thresh=args.conf_thresh,
                    nms_thresh=args.nms_thresh)

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
