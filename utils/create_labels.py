import numpy as np
import torch


def label_creator(targets,
                  num_classes,
                  fmp_size, 
                  strides=[8, 16, 32],
                  scale_range=[(0, 64), (64, 128), (128, 1e10)]):
    """
        targets: (list of tensors) annotations
        num_classes: (int) the number of class
        fmp_size: (list[int]) the shape [H, W] of each feature map
        stride: (list[int]) output stride of network
        scale_range: (list)
    """
    # prepare
    batch_size = len(targets)
    target_tensor = []
    for (fmp_h, fmp_w) in (fmp_size):
        # [B, H, W, cls + box + ctn + pos]
        target_tensor.append(np.zeros([batch_size, fmp_h, fmp_w, num_classes + 4 + 1 + 1]))

    # generate gt datas  
    for bi in range(batch_size):
        target_i = targets[bi]
        boxes_i = target_i["boxes"].numpy()
        labels_i = target_i["labels"].numpy()

        for box, label in zip(boxes_i, labels_i):
            cls_id = int(label)
            x1, y1, x2, y2 = box
            # center point
            xc = (x1 + x2) * 0.5
            yc = (y1 + y2) * 0.5

            for si, stride in enumerate(strides):
                sr = scale_range[si]
                # map the origin coords to feature map
                x1_s, x2_s = x1 / stride, x2 / stride
                y1_s, y2_s = y1 / stride, y2 / stride
                xc_s = xc / stride
                yc_s = yc / stride

                gridx = int(xc_s)
                gridy = int(yc_s)

                # By default, we only consider the 3x3 neighborhood of the center point
                for i in range(gridx - 1, gridx + 2):
                    for j in range(gridy - 1, gridy + 2):
                        if (j >= 0 and j < target_tensor[si].shape[1]) and (i >= 0 and i < target_tensor[si].shape[2]):
                            t = j - y1_s
                            b = y2_s - j
                            l = i - x1_s
                            r = x2_s - i
                            if min(t, b, l, r) > 0:
                                max_d= max(t, b, l, r)
                                left_p = sr[0] / stride
                                right_p = sr[1] / stride
                                if max_d >= left_p and max_d < right_p:
                                    ctn_label = np.sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
                                    box_label = np.array([x1_s, y1_s, x2_s, y2_s])
                                    # assignment
                                    target_tensor[si][bi, j, i, :num_classes] = 0.0 # avoiding the multi labels for one grid cell
                                    target_tensor[si][bi, j, i, cls_id] = 1.0
                                    target_tensor[si][bi, j, i, num_classes:num_classes + 4] = box_label
                                    target_tensor[si][bi, j, i, num_classes + 4] = ctn_label
                                    target_tensor[si][bi, j, i, -1] = 1.0


    # [B, N, cls + box + ctn + pos]
    target_tensor = [tgt.reshape(batch_size, -1, num_classes + 4 + 1 + 1) for tgt in target_tensor]
    target_tensor = np.concatenate(target_tensor, axis=1)
    
    return torch.from_numpy(target_tensor).float()


if __name__ == "__main__":
    pass
