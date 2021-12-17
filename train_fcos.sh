python train.py \
        --cuda \
        --root /home/jxk/object-detection/dataset/ \
        -d coco \
        -m fcos \
        -mc fcos_r50_fpn_1x \
        --lr 0.01 \
        --batch_size 8 \
        --train_min_size 800 \
        --train_max_size 1333 \
        --val_min_size 800 \
        --val_max_size 1333 \
        --wp_iter 500 \
        --optim adamw \
        --accumulate 2
