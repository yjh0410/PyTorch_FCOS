import numpy as np
import torch
import torch.nn as nn
from .conv import Conv
from .resnet import build_backbone
from .fpn import build_fpn


class FCOS_RT(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False):
        super(FCOS_RT, self).__init__()
        self.device = device
        self.fmp_size = None
        self.topk = 1000
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]

        # backbone
        self.backbone, feature_channels = build_backbone(model_name=cfg['backbone'],
                                                         pretrained=trainable,
                                                         train_backbone=True,
                                                         return_interm_layers=True)

        # neck
        self.neck = build_fpn(model_name=cfg['fpn'], in_channels=feature_channels, out_channel=cfg['head_dims'])

        # head
        self.cls_feat = nn.Sequential(
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm']),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm']),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm']),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm'])
        )
        self.reg_feat = nn.Sequential(
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm']),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm']),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm']),
            Conv(cfg['head_dims'], cfg['head_dims'], k=3, p=1, s=1, norm=cfg['norm'])
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


    def decode_boxes(self, fmp_size, reg_pred, stride=None):
        """
            fmp_size: (list) [fmp_h, fmp_w] the size of feature map
            reg_pred: (tensor) [HW, 4]
        """
        fmp_h, fmp_w = fmp_size
        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float() + 0.5  # [H, W, 2]
        anchor_xy = anchor_xy.to(self.device).view(-1, 2)  # [HW, 2]

        # decode box
        x1y1_pred = anchor_xy - reg_pred[..., :2].exp()
        x2y2_pred = anchor_xy + reg_pred[..., 2:].exp()
        bboxes = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return bboxes if stride is None else bboxes * stride


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

        # head
        outputs = {
            "scores": [],
            "labels": [],
            "bboxes": []
        }
        for i, p in enumerate(features):
            fmp_h_i, fmp_w_i = p.shape[-2:]
            cls_feat_i = self.cls_feat(p)
            reg_feat_i = self.reg_feat(p)
            # pred
            cls_pred_i = self.cls_pred(cls_feat_i)
            reg_pred_i = self.reg_pred(reg_feat_i)
            ctn_pred_i = self.ctn_pred(reg_feat_i)

            # [1, C, H, W] -> [H, W, C] -> [HW, C]
            cls_pred_i = cls_pred_i.permute(0, 2, 3, 1).contiguous()[0].view(-1, self.num_classes)
            reg_pred_i = reg_pred_i.permute(0, 2, 3, 1).contiguous()[0].view(-1, 4)
            ctn_pred_i = ctn_pred_i.permute(0, 2, 3, 1).contiguous()[0].view(-1, 1)

            # decode box
            bboxes_i = self.decode_boxes(fmp_size=[fmp_h_i, fmp_w_i], 
                                         reg_pred=reg_pred_i, 
                                         stride=self.stride[i])
            # normalize bbox
            bboxes_i[..., [0, 2]] /= img_w
            bboxes_i[..., [1, 3]] /= img_h
            bboxes_i = bboxes_i.clamp(0., 1.)

            # score
            scores_i = torch.sqrt(cls_pred_i.sigmoid() *  ctn_pred_i.sigmoid()).view(-1, self.num_classes)
            cls_scores_i, cls_inds_i = torch.max(scores_i, dim=-1)
            # topk
            if scores_i.shape[0] > self.topk:
                cls_scores_i, topk_scores_indx = torch.topk(cls_scores_i, self.topk)
                cls_inds_i = cls_inds_i[topk_scores_indx]
                bboxes_i = bboxes_i[topk_scores_indx]

            outputs["scores"].append(cls_scores_i)
            outputs["labels"].append(cls_inds_i)
            outputs["bboxes"].append(bboxes_i)
            
        outputs["scores"] = torch.cat(outputs["scores"], dim=0)      # [N,]
        outputs["labels"] = torch.cat(outputs["labels"], dim=0)      # [N,]
        outputs["bboxes"] = torch.cat(outputs["bboxes"], dim=0)      # [N, 4]

        # to cpu
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        bboxes = outputs["bboxes"].cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        return bboxes, scores, labels


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

            # neck: P3, P4, P5
            features = self.neck(x)

            # head
            outputs = {
                "pred_cls": [],
                "pred_box": [],
                "pred_ctn": [],
                "masks": [],
                "fmp_size": [],
                "strides": []
            }
            for i, p in enumerate(features):
                fmp_h_i, fmp_w_i = p.shape[-2:]
                cls_feat_i = self.cls_feat(p)
                reg_feat_i = self.reg_feat(p)
                # pred
                cls_pred_i = self.cls_pred(cls_feat_i)
                reg_pred_i = self.reg_pred(reg_feat_i)
                ctn_pred_i = self.ctn_pred(reg_feat_i)

                # [B, C, H, W] -> [B, H, W, C]
                cls_pred_i = cls_pred_i.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred_i = reg_pred_i.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                ctn_pred_i = ctn_pred_i.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            
                # decode box
                box_pred_i = self.decode_boxes(fmp_size=[fmp_h_i, fmp_w_i],
                                               reg_pred=reg_pred_i)

                outputs["pred_cls"].append(cls_pred_i)
                outputs["pred_box"].append(box_pred_i)
                outputs["pred_ctn"].append(ctn_pred_i)
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
