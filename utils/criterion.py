import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import giou_score
from .create_labels import label_creator
from utils.vis import vis_targets


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, mask=None):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                     target=targets, 
                                                     reduction="none"
                                                     )
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)
        loss = loss if mask is None else loss * mask[..., None]

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            pos_inds = (targets == 1.0).float()
            # scale loss by number of positive samples of each sample and batch size
            # [B, HW, C] -> [B,]
            batch_size = logits.size(0)
            num_pos = pos_inds.sum([1, 2]).clamp(1.0)
            loss = loss.sum([1, 2]) / num_pos
            loss = loss.sum() / batch_size

            # scale loss by number of total positive samples
            # num_pos = pos_inds.sum()
            # loss = loss.sum() / num_pos

        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class Criterion(nn.Module):
    def __init__(self, cfg, device, loss_cls_weight=1.0, loss_reg_weight=1.0, loss_ctn_weight=1.0, num_classes=80):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        self.loss_ctn_weight = loss_ctn_weight

        self.cls_loss_f = FocalWithLogitsLoss(reduction='mean')
        self.ctn_loss_f = nn.BCEWithLogitsLoss(reduction='none')


    def loss_labels(self, pred_cls, target, mask=None):
        # groundtruth    
        target_labels = target[..., :self.num_classes].float() # [B, HW, C]

        # cls loss
        loss_cls = self.cls_loss_f(pred_cls, target_labels, mask)

        return loss_cls


    def loss_bboxes(self, pred_box, target, mask=None):
        # groundtruth    
        target_bboxes = target[..., self.num_classes:self.num_classes+4] # [B, HW, 4]
        target_pos = target[..., -1].float()                             # [B, HW,]
        num_pos = target_pos.sum(-1).clamp(1.0)                         # [B,]

        # reg loss
        B, HW, _ = pred_box.size()
        # decode bbox: [B, HW, 4] -> [B x HW, 4]
        x1y1x2y2_pred = pred_box.view(-1, 4)
        x1y1x2y2_gt = target_bboxes.view(-1, 4)

        # giou: [B x HW,]
        pred_giou = giou_score(x1y1x2y2_pred, x1y1x2y2_gt)
        # [B x HW,] -> [B, HW,]
        pred_giou = pred_giou.view(B, HW)
        loss_reg = 1. - pred_giou if mask is None else (1. - pred_giou) * mask
        loss_reg = loss_reg * target_pos

        # scale loss by number of positive samples of each sample and batch size
        # [B, HW,] -> [B,]
        batch_size = pred_box.size(0)
        loss_reg = loss_reg.sum(-1) / num_pos
        loss_reg = loss_reg.sum() / batch_size

        # scale loss by number of total positive samples
        # num_pos = target_pos.sum()
        # loss_reg = loss_reg.sum() / num_pos
        
        return loss_reg


    def loss_centerness(self, pred_ctn, target, mask=None):
        # groundtruth    
        target_ctn = target[..., -2]                                     # [B, HW,]
        target_pos = target[..., -1].float()                             # [B, HW,]
        num_pos = target_pos.sum(-1).clamp(1.0)                          # [B,]

        # reg loss
        B, HW, _ = pred_ctn.size()

        # [B x HW,] -> [B, HW,]
        loss_ctn = self.ctn_loss_f(pred_ctn[..., 0], target_ctn)
        loss_ctn = loss_ctn if mask is None else loss_ctn * mask
        loss_ctn = loss_ctn * target_pos

        # scale loss by number of positive samples of each sample and batch size
        # [B, HW,] -> [B,]
        batch_size = pred_ctn.size(0)
        loss_ctn = loss_ctn.sum(-1) / num_pos
        loss_ctn = loss_ctn.sum() / batch_size
        
        # scale loss by number of total positive samples
        # num_pos = target_pos.sum()
        # loss_ctn = loss_ctn.sum() / num_pos

        return loss_ctn


    def forward(self, outputs, targets, images=None, vis_labels=False):
        """
            outputs["pred_cls"]: (tensor) [B, HW, C]
            outputs["pred_giou"]: (tensor) [B, HW, 1]
            outputs["masks"]: (tensor) [B, HW]
            outputs["fmp_size"]: (list) [H, W]
            outputs["strides"]: (list[int])
            target: (list) a list of annotations
            images: (tensor) [B, 3, H, W]
            vis_labels: (bool) visualize labels to check positive samples
        """
        batch_size = outputs["pred_cls"].size(0)
        # make labels
        targets = label_creator(targets=targets, 
                                num_classes=self.num_classes,
                                fmp_size=outputs["fmp_size"],
                                strides=outputs["strides"],
                                scale_range=self.cfg["scale_range"])

        # vis labels
        if vis_labels:
            vis_targets(images, targets, fmp_size=outputs["fmp_size"], strides=outputs["strides"])

        # [B, HW, C+4+1]
        targets = targets.to(self.device)

        # compute class loss
        loss_labels = self.loss_labels(outputs["pred_cls"], targets, outputs["masks"])

        # compute bboxes loss
        loss_bboxes = self.loss_bboxes(outputs["pred_box"], targets, outputs["masks"])

        # compute centerness loss
        loss_centerness = self.loss_centerness(outputs["pred_ctn"], targets, outputs["masks"])

        # total loss
        losses = self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes + \
                 self.loss_ctn_weight * loss_centerness

        return loss_labels, loss_bboxes, loss_centerness, losses


def build_criterion(args, cfg, device, num_classes=80):
    criterion = Criterion(cfg=cfg,
                          device=device,
                          loss_cls_weight=args.loss_cls_weight,
                          loss_reg_weight=args.loss_reg_weight,
                          loss_ctn_weight=args.loss_ctn_weight,
                          num_classes=num_classes)
    return criterion

    
if __name__ == "__main__":
    pass
