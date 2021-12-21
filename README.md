# PyTorch_FCOS
A PyTorch version of FCOS.

I currently do not have enough resources to debug this project ...

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n fcos python=3.6
```

- Then, activate the environment:
```Shell
conda activate fcos
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.1.0 and Torchvision >= 0.3.0

# Experiments
On COCO-val

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> Model </th><td bgcolor=white> Size </td><td bgcolor=white> schedule </td><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8>     FCOS      </th><td bgcolor=white> (800, 1333) </td><td bgcolor=white> 12 epoch </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> FCOS-RT-FPN   </th><td bgcolor=white> (512, 736) </td><td bgcolor=white> 48 epoch </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> FCOS-RT-PAN   </th><td bgcolor=white> (512, 736) </td><td bgcolor=white> 48 epoch </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> FCOS-RT-BiFPM </th><td bgcolor=white> (512, 736) </td><td bgcolor=white> 48 epoch </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

</table></tbody>

# Train
## Visualize positive sample
You can run following command to visualize positiva sample:
```Shell
python train.py \
        -d voc \
        --batch_size 2 \
        --root path/to/your/dataset \
        --vis_targets
```

## Train FCOS and FCOS-RT
```Shell
sh train_fcos.sh
```

```Shell
sh train_fcosrt.sh
```

Attention, you need to change th`--root` in the `train_fcos.sh` or `train_fcosrt.sh` to your path to dataset.

For more configuration details of FCOS and FCOS-RT, you should open the `config/fcos_config.py` to check.

# Test
For example:

```Shell
python test.py -d coco \
               --root path/to/dataset/ \
               --cuda \
               --test_min_size 512 \
               --test_max_size 736 \
               -m fcos_rt \
               -mc fcos_rt_r50_fpn_4x\
               --weight path/to/weight \
               -vs 0.5 \
               --show
```

# Evaluation
For example:

```Shell
python eval.py -d coco-val \
               --root path/to/dataset/ \
               --cuda \
               --test_min_size 512 \
               --test_max_size 736 \
               -m fcos_rt \
               -mc fcos_rt_r50_fpn_4x\
               --weight path/to/weight
```

# Evaluation on COCO-test-dev
Attentio, you must be sure that you have downloaded test2017 split of COCO.

For example:

```Shell
python eval.py -d coco-test \
               --root path/to/dataset/ \
               --cuda \
               --test_min_size 512 \
               --test_max_size 736 \
               -m fcos_rt \
               -mc fcos_rt_r50_fpn_4x\
               --weight path/to/weight
```

You will get a `coco_test-dev.json` file. 
Then you should follow the official requirements to compress it into zip format 
and upload it the official evaluation server.

# Reference
Paper: 

```Shell
@article{tian2020fcos,
  title={Fcos: A simple and strong anchor-free object detector},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```