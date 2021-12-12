import numpy as np
import cv2


def vis_data(images, targets, masks):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
        masks: (tensor) [B, H, W]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image = image.copy()

        boxes = targets[bi]["boxes"]
        labels = targets[bi]["labels"]
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.imshow('groundtruth', image)
        cv2.waitKey(0)

        # to numpy
        mask = masks[bi].cpu().numpy()
        mask = (mask * 255).astype(np.uint8).copy()

        boxes = targets[bi]["boxes"]
        labels = targets[bi]["labels"]
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)


def vis_targets(images, targets, fmp_size, strides=[8, 16, 32]):
    """
        images: (tensor) [B, 3, H, W]
        targets: (tensor) [B, HW, C+4+1]
        fmp_size: (list) [H, W]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image = image.copy()

        target_i = targets[bi] # [HW, C+4+1+1]
        start_grid = 0
        for size, stride in zip(fmp_size, strides):
            for j in range(size[0]):
                for i in range(size[1]):
                    grid_i = start_grid + j * size[1] + i
                    target_ii = target_i[grid_i]
                    if target_ii[-1].item() > 0.0: # positive sample
                        # gt box
                        box = target_ii[-6:-2]
                        x1s, y1s, x2s, y2s = box
                        x1 = int(x1s * stride)
                        y1 = int(y1s * stride)
                        x2 = int(x2s * stride)
                        y2 = int(y2s * stride)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # anchor point
                        x = int((i + 0.5) * stride)
                        y = int((j + 0.5) * stride)
                        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            start_grid += size[0] * size[1]

        cv2.imshow('groundtruth', image)
        cv2.waitKey(0)
