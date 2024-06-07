
import json
import os
from typing import List
import copy
from functools import partial
import random
import numpy as np
import torch
import logging
from einops import rearrange
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import VIS_TrainMapper, VIS_EvalMapper
from .vis_frame_sampler import VIS_FRAMES_SAMPLER_REGISTRY
from data_schedule.vis.apis import VIS_Dataset, VIS_Aug_CallbackAPI,\
    VIS_TrainAPI_clipped_video, VIS_EvalAPI_clipped_video_request_ann

@MAPPER_REGISTRY.register()
class VIS_Video_EvalMapper(VIS_EvalMapper):
    def __init__(self,
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,
                 ): 
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('step_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict):
        VIS_Dataset
        video_id, all_frames = data_dict['video_id'], data_dict['all_frames']
        video_frames = self.get_frames_fn(video_id=video_id, frames=all_frames)
        aug_ret = {
            'video': video_frames,
            'callback_fns': []
        }
        VIS_Aug_CallbackAPI
        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')
        callback_fns = aug_ret.pop('callback_fns')[::-1]
        VIS_EvalAPI_clipped_video_request_ann
        return {
            'video_dict': {'video': video},
            'meta': {
                'video_id': video_id,
                'frames': all_frames,
                'request_ann': torch.ones(len(all_frames)).bool(),
                'callback_fns': callback_fns               
            }
        }

@MAPPER_REGISTRY.register()
class VIS_Video_or_Step_To_Clip_TrainMapper(VIS_TrainMapper):
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        
        self.frames_sampler = VIS_FRAMES_SAMPLER_REGISTRY.get(\
            mapper_config['frames_sampler']['name'])(sampler_configs=mapper_config['frames_sampler'],
                                                    dataset_meta=dataset_meta)

    def _call(self, data_dict):
        VIS_Dataset
        video_id, all_frames, all_objs = data_dict['video_id'], data_dict['all_frames'], data_dict['all_objs']
        frame_idx = data_dict['frame_idx'] if 'frame_idx' in data_dict else None

        all_obj_ids = list(all_objs.keys()) # [1, 2, 5, 4]
        assert len(list(set(all_obj_ids))) == len(all_obj_ids)
        class_labels = torch.tensor([all_objs[key]['class_label'] for key in all_obj_ids]) # [8, 10, 20 34]

        re_sample = True
        sampled_counts = 0
        while re_sample:
            sampled_frames = self.frames_sampler(all_frames=all_frames, frame_idx=frame_idx, video_id=video_id)
            # t' h w, int, obj_ids ;  has_ann t
            frames_mask, has_ann = self.get_frames_mask_fn(video_id=video_id, frames=sampled_frames)
            appear_objs = frames_mask.unique() # [0, 1, 2]
            assert set(appear_objs.tolist()).issubset(set([0] + all_obj_ids))
            re_sample = (len(list(set(appear_objs.tolist()) & set(all_obj_ids))) == 0)
            # 只要出现某些个物体就行
            sampled_counts += 1
            if sampled_counts > 2:
                logging.error('sampled two much times')
                raise RuntimeError()
            
        frames_mask = torch.stack([frames_mask == obj_id for obj_id in all_obj_ids], dim=0) # N t' h w, bool
        video_frames = self.get_frames_fn(video_id=video_id, frames=sampled_frames) 
        width, height = video_frames[0].size
        aug_ret = {
            'video': video_frames,
            'masks': frames_mask, # N t' h w
            'has_ann': has_ann, # t
            'classes': class_labels, # N
        }
        VIS_Aug_CallbackAPI
        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')
        frame_targets = self.map_to_frame_targets(aug_ret)
        if self.clip_global_targets_map_to_local_targets:
            aug_ret = self.map_global_targets_to_local_targets(aug_ret)

        VIS_TrainAPI_clipped_video
        ret = {}
        ret['video_dict'] = {'video': video}
        ret['targets'] = aug_ret
        ret['frame_targets'] = frame_targets
        return ret








