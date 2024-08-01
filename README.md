
## LGRNet: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos 
Huihui Xu, Yijun Yang(Collect UFUV data), Angelica Aviles-Rivero, Guang Yang, Jing Qin, and Lei Zhu*


[`Paper`](https://arxiv.org/abs/2407.05703) | [`BibTeX`](#citing) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lgrnet-local-global-reciprocal-network-for/video-polyp-segmentation-on-sun-seg-easy-1)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-easy-1?p=lgrnet-local-global-reciprocal-network-for)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lgrnet-local-global-reciprocal-network-for/video-polyp-segmentation-on-sun-seg-hard-1)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-hard-1?p=lgrnet-local-global-reciprocal-network-for)


This is the official implmentation of LGRNet (MICCAI'24 Early Accept), which incorporates local **[Cyclic Neighborhoold Propagation](https://github.com/bio-mlhui/LGRNet/blob/main/models/encoder/neighborhood_qk.py#L57)** and global **[Hilbert Selective Scan](https://github.com/bio-mlhui/LGRNet/blob/main/models/encoder/ops/modules/frame_query_ss2d.py#L576)**. Together with the notion of **[Frame Bottleneck Queries](https://github.com/bio-mlhui/LGRNet/blob/main/models/encoder/localGlobal.py#L185)**, LGRNet can both efficiently and effectively aggregate the local-global temporal context, which achieves state-of-the-art on the public [Video Polyp Segmentation(VPS)](https://paperswithcode.com/task/video-polyp-segmentation) benchmark.

<div align="justify">As an example for ultrasound video, a single frame is too noisy and insufficient for accurate lesion diagnosis. In practice, doctors need to check neighboring frames(local) and collect all visual clues (global) in the video to predict possible lesion region and filter out irrelevent surrounding issues </div>
</br>
<div align="center" style="padding: 0 100pt">
<img src="assets/images/pipeline.png">
</div>
</br>



## Getting started

1. Installation: Please refer to [INSTALL.md](assets/INSTALL.md) for more details.
2. Data preparation: Please refer to [DATA.md](assets/DATA.md) for more details.
3. Training: Please refer to [TRAIN.md](assets/TRAIN.md) for more details.
4. Testing: Please refer to [TEST.md](assets/TEST.md) for more details. 
5. Model zoo: Please refer to [MODEL_ZOO.md](assets/MODEL_ZOO.md) for more details.

# Introduction 

LGRNet consists of a backbone, a spatial-temporal encoder, and a Mask2Former decoder, as illustrated in Figure. Each encoder layer is composed of a CNP layer and HilbertSS layer, which aggregates temporal context in a local-global manner.


The CNP layer is a temporal extensioin of Neighborhood Attention. We also add local cyclic inter-frame dependency constraint for more efficient propagation.
![CNP](assets/images/cnp.png)

The HilberSS layer takes the highly-semantic frame bottleneck queries as input, selectively scans the queries in Hilbert manner to enable global temporal context propagation. The global context is then cross-attended to each frame feature maps for local refinement, which forms a reciprocal net.
![HilbertSS](assets/images/hilbert.png)


Based on the above designs, LGRNet can be used as a baseline for binary or multi-semantic Medical Video Object Segmentation.



# Results

## UFUV

![UFUV](/assets/images/ufuv.png)

## CVCs

![CVC](/assets/images/cvcs.png)

## SUN-SEG

![SUN-SEG](/assets/images/sunseg.png)

# Citing LGRNet

<!-- ```
@misc{wu2023GLEE,
  author= {Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, Song Bai},
  title = {General Object Foundation Model for Images and Videos at Scale},
  year={2023},
  eprint={2312.09158},
  archivePrefix={arXiv}
}
``` -->

## Acknowledgments

- Thanks [Gilbert](https://github.com/jakubcerveny/gilbert) for the implementation of Hilbert curve generation.
<!-- 
- Thanks [Natten](https://github.com/SHI-Labs/NATTEN) for Neighborhood Attention.

- Thanks [Mamba](https://github.com/state-spaces/mamba) for S6.

- Thanks [VMamba](https://github.com/MzeroMiko/VMamba) for SS2D.

- Thanks [Mask2Former](https://github.com/jakubcerveny/gilbert) for the implementation of Hilbert curve generation. --># LGRNet
