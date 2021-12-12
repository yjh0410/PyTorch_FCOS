python train.py \
        --cuda \
        -d coco \
        -v fcos_rt_r50_fpn_C5_1x \
        --batch_size 16 \
        --img_size 512 \
        --lr 0.01 \
        --wp_iter 500 \
        --accumulate 1
