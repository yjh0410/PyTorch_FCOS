python train.py \
        --cuda \
        -d coco \
        -v fcos_r50_fpn_1x \
        --lr 0.01 \
        --norm GN \
        --batch_size 16 \
        --img_size 800 \
        --wp_iter 500 \
        --accumulate 1
