# Install
## Requirements
We test the codes in the following environments

- CUDA 12.1
- Python 3.10.13
- Pytorch 2.1.1
- Torchvison 0.16.1
- detectron 0.6
- mamba_ssm 1.2.0.post1
- natten 0.15.1
- timm 0.9.12

## Install environment for LGRNet

```
conda create --name lgrnet python=3.10
conda activate lgrnet

# make sure CUDA-12.1 is installed and activated in env var.

# install torch
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121


# install detectron2, building may take much time.
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# install mamba, see https://github.com/state-spaces/mamba
cd .. 
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install . --no-build-isolation
cd ../LGRNet

# install natten, see https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md
pip install natten==0.15.1+torch210cu121 -f https://shi-labs.com/natten/wheels/cu121/torch2.1.0/natten-0.15.1%2Btorch210cu121-cp310-cp310-linux_x86_64.whl

# misc
pip install albumentations==1.3.1
pip install Pygments
pip install imgaug
pip install timm==0.9.12

# compile deform attention
cd models/encoder/ops/
python setup.py build install --user

# download resnet/pvtv2 ckpt, our model uses the same backbone with WeakPoly(phttps://github.com/weijun88/WeakPolyp)
wget -P ./pt/pvt_v2/pvt_v2_b2.pth  https://huggingface.co/huihuixu/lgrnet_ckpts/blob/main/pvt_v2_b2.pth  
wget -P ./pt/res2net/res2net50_v1b_26w_4s-3cf99910.pth  https://huggingface.co/huihuixu/lgrnet_ckpts/blob/main/res2net50_v1b_26w_4s-3cf99910.pth

```
