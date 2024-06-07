from detectron2.utils.registry import Registry
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from einops import rearrange, reduce, repeat
VIS_EVAL_AUG_REGISTRY = Registry('VIS_EVAL_AUG')
VIS_TRAIN_AUG_REGISTRY = Registry('VIS_TRAIN_AUG')


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)

def get_tgt_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
        return size[::-1]
    else:
        return get_size_with_aspect_ratio(image_size, size, max_size)

def pil_torch_to_numpy(video, masks, has_ann, float_image=True):
    # n t' h w
    # list[pil_image, rgb], t
    # t
    N, T = masks.shape[:2]
    has_ann_idx = torch.where(has_ann)[0] # time_idx
    # list[Image], t -> list[h w 3, 255uint8], t
    masks = masks.permute(1, 0, 2, 3).contiguous().unbind(0) # list[n h w] t'
    numpy_masks = [[]] * len(has_ann) # list[list[h w, 01_uint8], n], t
    assert len(has_ann_idx) == len(masks)
    for fmask, taylor in zip(masks, has_ann_idx): # n h w
        fnumpy_masks = []
        for mk in fmask.unbind(0): # h w
            fnumpy_masks.append(mk.numpy().astype(np.uint8))
        numpy_masks[taylor] = fnumpy_masks
    
    if float_image:
        # list[h w 3, 0-1float], t
        video = [F.to_tensor(frame).permute(1,2,0).numpy() for frame in video]
    else:
        # uint8
        video = [np.array(frame) for frame in video]
    
    return video, numpy_masks

def numpy_to_pil_torch(video, masks, has_ann):
    # numpy, numpy -> torch, torch
    # list[h w 3, 0-1float], t
    H, W = video[0].shape[:2]
    T = has_ann.int().sum()
    video = [Image.fromarray(np.uint8(aug_vid * 255), mode="RGB")  for aug_vid in video]
    # t'n h w
    torch_masks = torch.stack([torch.from_numpy(obj_mk).bool() for frame_mk in masks for obj_mk in frame_mk], dim=0)
    torch_masks = rearrange(torch_masks, '(T N) h w -> N T h w', T=T) # n t' h w

    return video, torch_masks 







