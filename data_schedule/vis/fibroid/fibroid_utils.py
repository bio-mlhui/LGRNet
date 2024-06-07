import wandb
import plotly.express as px
import logging
import os
import numpy as np
import torch
import json
from joblib import Parallel, delayed
import multiprocessing
import torch.distributed as dist
import detectron2.utils.comm as comm

import pycocotools.mask as mask_util
from pycocotools.mask import encode, area

from data_schedule.utils.segmentation import bounding_box_from_mask
from data_schedule.utils.video_clips import generate_windows_of_video
from glob import glob
from PIL import Image

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.png'),).convert('RGB') for f in frames]

# t' h w, int, obj_ids ;  has_ann t
def get_frames_mask(mask_path, video_id, frames):
    masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    masks = (masks > 0).int()
    return masks, torch.ones(len(frames)).bool()


SET_NAME = [
        'fibroid_train', 
        'fibroid_validate',
        'weakpolyp_train',

        'fibroid_validate_temp7',
        'fibroid_train_temp7',
        # 'weakpolyp_fibroid_train_temp7',

        'fibroid_validate_temp8',
        'fibroid_train_temp8',
        'weakpolyp_fibroid_train_temp8'
         ]

SET_NAME_TO_DIR = {
    'fibroid_train': 'temp/train',
    'fibroid_validate': 'temp/test',
    'weakpolyp_train': 'temp/uterus_myoma_WeakPolyP_temp/train',


    'fibroid_validate_temp7': 'temp7/test',
    'fibroid_train_temp7': 'temp7/train',
    'weakpolyp_fibroid_train_temp7': 'temp7/uterus_myoma_WeakPolyP_temp7/train',

    'fibroid_validate_temp8': 'temp8/test',
    'fibroid_train_temp8': 'temp8/train',
    'weakpolyp_fibroid_train_temp8': 'temp8/uterus_myoma_WeakPolyP_temp8/train',

}

SET_NAME_TO_NUM_VIDEOS = {
    'fibroid_train': 80,
    'fibroid_validate': 20,
    'weakpolyp_train': 80,


    'fibroid_train_temp7': 85,
    'fibroid_validate_temp7': 15,
    'weakpolyp_fibroid_train_temp7': 85 ,     


    'fibroid_train_temp8': 83,
    'fibroid_validate_temp8': 17,
    'weakpolyp_fibroid_train_temp8': 83     
}

SET_NAME_TO_MODE = {
    'fibroid_train': 'train',
    'fibroid_validate': 'evaluate',
    'weakpolyp_train': 'train',

    'fibroid_train_temp7': 'train',
    'fibroid_validate_temp7': 'evaluate',
    'weakpolyp_fibroid_train_temp7': 'train',   

    'fibroid_train_temp8': 'train',
    'fibroid_validate_temp8': 'evaluate',
    'weakpolyp_fibroid_train_temp8': 'train' 
}

SET_NAME_TO_PREFIX = {
    'fibroid_train': 'fibroid_train',
    'fibroid_validate': 'fibroid_validate',
    'weakpolyp_train': 'weakpolyp_fibroid_train',

    'fibroid_train_temp7': 'fibroid_train_temp7',
    'fibroid_validate_temp7': 'fibroid_validate_temp7',
    'weakpolyp_fibroid_train_temp7': 'weakpolyp_fibroid_train_temp7' ,

    'fibroid_train_temp8': 'fibroid_train_temp8',
    'fibroid_validate_temp8': 'fibroid_validate_temp8',
    'weakpolyp_fibroid_train_temp8': 'weakpolyp_fibroid_train_temp8' 

}

SET_NAME_TO_GT_TYPE = {
    'fibroid_train': 'GT',
    'fibroid_validate': 'GT',
    'weakpolyp_train': 'Box',

    'fibroid_train_temp7': 'GT',
    'fibroid_validate_temp7': 'GT',
    'weakpolyp_fibroid_train_temp7': 'Box', 

    'fibroid_train_temp8': 'GT',
    'fibroid_validate_temp8': 'GT',
    'weakpolyp_fibroid_train_temp8': 'Box' 
}
