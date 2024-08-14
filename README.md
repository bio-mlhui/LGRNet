
## LGRNet: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos [`Paper`](https://arxiv.org/abs/2407.05703) | [`BibTeX`](#citing) 
Huihui Xu, Yijun Yang(📈), Angelica Aviles-Rivero, Guang Yang, Jing Qin, and Lei Zhu

######  📈: collected UFUV data, UFUV may be made open-accessed according to their permission

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lgrnet-local-global-reciprocal-network-for/video-polyp-segmentation-on-sun-seg-hard)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-hard?p=lgrnet-local-global-reciprocal-network-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lgrnet-local-global-reciprocal-network-for/video-polyp-segmentation-on-sun-seg-easy)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-easy?p=lgrnet-local-global-reciprocal-network-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lgrnet-local-global-reciprocal-network-for/video-polyp-segmentation-on-sun-seg-easy-1)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-easy-1?p=lgrnet-local-global-reciprocal-network-for)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lgrnet-local-global-reciprocal-network-for/video-polyp-segmentation-on-sun-seg-hard-1)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-hard-1?p=lgrnet-local-global-reciprocal-network-for)



This is the official implmentation of LGRNet (MICCAI'24 Early Accept), which incorporates local **[Cyclic Neighborhoold Propagation](https://github.com/bio-mlhui/LGRNet/blob/main/models/encoder/neighborhood_qk.py#L57)** and global **[Hilbert Selective Scan](https://github.com/bio-mlhui/LGRNet/blob/main/models/encoder/ops/modules/frame_query_ss2d.py#L531)**. Together with the notion of **[Frame Bottleneck Queries](https://github.com/bio-mlhui/LGRNet/blob/main/models/encoder/localGlobal.py#L185)**, LGRNet can both efficiently and effectively aggregate the local-global temporal context, which achieves *state-of-the-art* on the public [Video Polyp Segmentation(VPS)](https://paperswithcode.com/task/video-polyp-segmentation) benchmark.

<div align="justify">As an example for ultrasound video, a single frame is too noisy and insufficient for accurate lesion diagnosis. In practice, doctors need to check neighboring frames(local) and collect all visual clues (global) in the video to predict possible lesion region and filter out irrelevent surrounding issues. </div>
</br>
<div align="center" style="padding: 0 100pt">
<img src="assets/images/pipeline.png">
</div>
</br>
<div align="justify"> In CNP, each token takes the neighborhood tokens (defined by a kernel) in the cyclic frame as attention keys. CNP enables aggregating the local(cyclic) temporal information into one token. In Hilbert Selective Scan, a set of frame bottleneck queries are used to aggreate spatial information from each frame. Then, we use Hilbert Selective Scan to efficiently parse the global temporal context based on these bottleneck queries. The global temporal context is then propagated back to the feature maps by a Distribute layer. Based on Mask2Former, the decoder can output a set of different mask predictions with corresponding confidence score, which also facilitates comprehesive diagnosis.</div>


## Items

1. Installation: Please refer to [INSTALL.md](assets/INSTALL.md) for more details.
2. Data preparation: Please refer to [DATA.md](assets/DATA.md) for more details.


3. Training: 

Change PORT_NUM for DDP and make sure the $CURRENT_TASK is 'VIS': (my framework is task-agnostic if you ever noticed 😃)
```
export CURRENT_TASK=VIS
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=PORT_NUM
```

Make sure the $PT_PATH and $DATASET_PATH are correctly set during installation and preparing data.

The training on SUN-SEG is conducted using 2 4090-24GB GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 TORCH_NUM_WORKERS=8 python main.py --config_file output/VIS/sunseg/pvt/pvt.py --trainer_mode train_attmpt
```

4. logs, checkpoints, predictions

| Backbone| Dataset | Dice | mIou  | log | ckpt | predictions |
| :----: | :----: | :----: | :----: | :----: | :----: |:----: |
| PVTv2-B2 | SUN-SEG-Train | -- | -- | [log](https://drive.google.com/file/d/17MTOYW73RLbvZS3BLFBZEphY_0JzN6er/view?usp=sharing) | [ckpt](https://drive.google.com/file/d/1D4YAIfFCCQIsDfKgSCr9tCw7vDAgqf76/view?usp=sharing) | --
| PVTv2-B2 | SUN-SEG-Hard-Testing | 0.876 | 0.805 | [log](https://drive.google.com/file/d/1wdVMWMknSlURaBROWbMax4iS9V1Tbn9-/view?usp=sharing) |[ckpt](https://drive.google.com/file/d/1D4YAIfFCCQIsDfKgSCr9tCw7vDAgqf76/view?usp=sharing) | [mask predictions](https://drive.google.com/file/d/1V8CDMC87o7t4eyts4BVEwflDUrFpAOVX/view?usp=sharing)
| PVTv2-B2 | SUN-SEG-Easy-Testing | 0.875 | 0.810 | [log](https://drive.google.com/file/d/1wdVMWMknSlURaBROWbMax4iS9V1Tbn9-/view?usp=sharing) |[ckpt](https://drive.google.com/file/d/1D4YAIfFCCQIsDfKgSCr9tCw7vDAgqf76/view?usp=sharing) | [mask predictions](https://drive.google.com/file/d/1V8CDMC87o7t4eyts4BVEwflDUrFpAOVX/view?usp=sharing)
| PVTv2-B2 | SUN-SEG-Hard-Unseen-Testing | 0.865 | 0.792 | [log](https://drive.google.com/file/d/1obt_qvWCvslhRY-e4SrTJNS0r6Diad4e/view?usp=sharing) | [mask predictions](https://drive.google.com/file/d/1V8CDMC87o7t4eyts4BVEwflDUrFpAOVX/view?usp=sharing)
| PVTv2-B2 | SUN-SEG-Easy-Unseen-Testing | 0.853 | 0.783 | [log](https://drive.google.com/file/d/1obt_qvWCvslhRY-e4SrTJNS0r6Diad4e/view?usp=sharing) | [mask predictions](https://drive.google.com/file/d/1V8CDMC87o7t4eyts4BVEwflDUrFpAOVX/view?usp=sharing)
| Res2Net-50 | SUN-SEG-Hard-Testing | 0.841 | 0.765 | [log](https://drive.google.com/file/d/17pUxFMuHpPD_In5RVrJUsPFZGOgNFzb6/view?usp=sharing) |
| Res2Net-50 | SUN-SEG-Easy-Testing | 0.843 | 0.774 | [log](https://drive.google.com/file/d/17pUxFMuHpPD_In5RVrJUsPFZGOgNFzb6/view?usp=sharing) |
| PVTv2-B2 | CVC612V | 0.933 | 0.877 | [log](https://drive.google.com/file/d/1m36mJL0Fu3T9F73TqFGnFsWaCGh3JDeJ/view?usp=drive_link) |
| PVTv2-B2 | CVC300TV | 0.916 | 0.852 | [log](https://drive.google.com/file/d/1m36mJL0Fu3T9F73TqFGnFsWaCGh3JDeJ/view?usp=drive_link) |
| PVTv2-B2 | CVC612T | 0.875 | 0.814 | [log](https://drive.google.com/file/d/1m36mJL0Fu3T9F73TqFGnFsWaCGh3JDeJ/view?usp=drive_link) |


5. Evaluate:
Evaluating on SUN-SEG-Easy AND SUN-SEG-Hard using 1 4090-24GPU GPUS (**modify the ckpt_path to the absolute path**):
```
CUDA_VISIBLE_DEVICES=0 TORCH_NUM_WORKERS=8 python main.py --config_file output/VIS/sunseg/pvt/pvt.py --trainer_mode eval --eval_path /home/xuhuihui/workspace/LGRNet/output/VIS/sunseg/pvt/epc[0_00]_iter[0]_sap[0]/ckpt.pth.tar 
```

## citing
```
@article{xu2024lgrnet,
  title={LGRNet: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos},
  author={Xu, Huihui and Yang, Yijun and Aviles-Rivero, Angelica I and Yang, Guang and Qin, Jing and Zhu, Lei},
  journal={arXiv preprint arXiv:2407.05703},
  year={2024}
}
``` 

## Acknowledgments
- Thanks [Gilbert](https://github.com/jakubcerveny/gilbert) for the implementation of Hilbert curve generation.
- Thanks GPT4 for helping me constructing idea of Hilbert Filling Curve v.s. Zigzag curve