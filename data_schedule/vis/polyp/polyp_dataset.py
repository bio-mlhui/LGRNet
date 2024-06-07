from typing import Optional, Union
import json
import os
from functools import partial
import numpy as np
import torch
import logging
from tqdm import tqdm
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog
from collections import defaultdict
from .polyp_utils import get_frames, get_frames_mask, SET_NAME_TO_DIR, SET_NAME, SET_NAME_TO_NUM_VIDEOS, SET_NAME_TO_MODE, SET_NAME_TO_PREFIX

def polyp_train(step_size,
                split_dataset_name,
                video_ids,
                video_to_frames,
                root_path):

    logging.debug(f'{split_dataset_name} Generating metas...')   
    metas = []
    for vid_id in tqdm(video_ids):
        all_frames = sorted(video_to_frames[vid_id])
        poly_class = 0
        if step_size is None:  
            metas.append({
                'video_id': vid_id,
                'all_frames' : all_frames,
                'all_objs': { 1: {'class_label': poly_class} },
                'meta_idx': len(metas)
            }) 
        else:
            for frame_idx in range(0, len(all_frames), step_size):
                metas.append({
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'all_objs': { 1: {'class_label': poly_class} },
                    'meta_idx': len(metas)
                })                

    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')
    return metas


def polyp_evaluate(eval_video_ids,
                   split_dataset_name,
                   step_size,
                   video_to_frames):
    if (step_size is not None) and (step_size > 1):
        logging.warning('why?')
        raise ValueError()
    metas = []
    for video_id in eval_video_ids:
        all_frames = sorted(video_to_frames[video_id])
        if step_size == None:
            metas.append({
                'video_id': video_id,
                'all_frames': all_frames,
                'meta_idx': len(metas)
            })     
    
        else:   
            for frame_idx in range(0, len(all_frames), step_size):
                metas.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'meta_idx': len(metas)
                })                                 

    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')  
    return metas

_root = os.getenv('DATASET_PATH')

root = os.path.join(_root, 'SUN/SUN-SEG2')

visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['polyp_train_step[6]'] = [] 
visualize_meta_idxs['polyp_train'] = [] 
visualize_meta_idxs['polyp_hard_unseen'] = []
visualize_meta_idxs['polyp_hard_seen'] = []
visualize_meta_idxs['polyp_easy_unseen'] = []
visualize_meta_idxs['polyp_easy_seen'] = []

polyp_meta = {
    'thing_classes': ['polyp', 'not polyp'],
    'thing_colors': [(255., 140., 0.), (0., 255., 0.)],
    'root': root
}

for name in SET_NAME:
    set_dir = SET_NAME_TO_DIR[name]
    set_dir = os.path.join(root, set_dir)
    num_videos = SET_NAME_TO_NUM_VIDEOS[name]

    video_ids = os.listdir(os.path.join(set_dir, 'Frame'))
    assert len(video_ids) == num_videos

    video_to_frames = {
        vid: sorted([png[:-4] for png in os.listdir(os.path.join(set_dir, 'Frame', vid)) if png.endswith('.jpg')])\
            for vid in video_ids
    }
    mode = SET_NAME_TO_MODE[name]
    if mode == 'train':
        prefix = SET_NAME_TO_PREFIX[name]
        train_meta = copy.deepcopy(polyp_meta)
        train_meta.update({
            'mode': 'train',
            'get_frames_fn': partial(get_frames, frames_path=os.path.join(set_dir, 'Frame')),
            'get_frames_mask_fn': partial(get_frames_mask, mask_path=os.path.join(set_dir, 'GT'),),
        })

        # train
        for step_size in [1, 3, 6, 9, 12, None]:
            step_identifer = '' if step_size is None else f'_step[{step_size}]'
            split_name = f'{prefix}{step_identifer}'
            train_meta.update({'name': split_name})
            DatasetCatalog.register(split_name, partial(polyp_train,
                                                        video_ids=video_ids, 
                                                        split_dataset_name=split_name,
                                                        step_size=step_size,
                                                        video_to_frames=video_to_frames,
                                                        root_path=set_dir))    
            MetadataCatalog.get(split_name).set(**train_meta, 
                                                step_size=step_size,
                                                visualize_meta_idxs=visualize_meta_idxs[split_name]) 
    elif mode == 'evaluate':
        prefix = SET_NAME_TO_PREFIX[name]
        validate_meta = copy.deepcopy(polyp_meta)

        validate_meta.update({
            'mode': 'evaluate',
            'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, os.path.join(set_dir, 'Frame'))),
            'eval_set_name': SET_NAME_TO_DIR[name],
            'get_frames_gt_mask_fn': partial(get_frames_mask, mask_path=os.path.join(root, os.path.join(set_dir, 'GT')),),
            'eval_meta_keys': video_to_frames
        })
        for step_size in  [1, None,]:
            step_identifer = '' if step_size is None else f'_step[{step_size}]'
            split_name = f'{prefix}{step_identifer}'
            validate_meta.update({'name': split_name})
            DatasetCatalog.register(split_name, partial(polyp_evaluate,
                                                        eval_video_ids=video_ids, 
                                                        split_dataset_name=split_name, 
                                                        step_size=step_size,
                                                        video_to_frames=video_to_frames))    
            MetadataCatalog.get(split_name).set(**validate_meta, step_size=step_size,
                                                visualize_meta_idxs=visualize_meta_idxs[split_name])
