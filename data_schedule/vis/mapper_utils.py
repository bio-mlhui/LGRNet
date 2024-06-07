from .vis_aug_utils import VIS_EVAL_AUG_REGISTRY, VIS_TRAIN_AUG_REGISTRY
import torch
from copy import deepcopy as dcopy
from data_schedule.registry import Mapper
import copy
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video

class VIS_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,) -> None:
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.get_frames_fn = dataset_meta.get('get_frames_fn')

class VIS_TrainMapper(VIS_Mapper):

    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')   
        self.clip_global_targets_map_to_local_targets = mapper_config['clip_global_targets_map_to_local_targets'] 
        self.augmentation = VIS_TRAIN_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])

    def map_to_frame_targets(self, clip_targets):
        VIS_TrainAPI_clipped_video
        clip_rets = copy.deepcopy(clip_targets)
        masks = clip_rets['masks'].transpose(0, 1).contiguous() # t' N h w
        class_labels = clip_rets['classes'] # [10, 32, 10, 4]
        has_box = 'boxes' in clip_rets
        if has_box:
            boxes = clip_rets['boxes'].transpose(0, 1).contiguous() # t' N 4
            assert len(masks) == len(boxes)
        ret = []
        for idx, frame_mk in enumerate(masks):
            frame_targets = {
                'masks': frame_mk.unsqueeze(1), # N 1 h w
                'classes': class_labels, # N
            }
            if has_box:
                frame_targets.update({'boxes': boxes[idx].unsqueeze(1)}) # N 1 4
            if self.clip_global_targets_map_to_local_targets:
                frame_targets = self.map_global_targets_to_local_targets(frame_targets)
            frame_targets['masks'] = frame_targets['masks'].squeeze(1)
            if has_box:
                frame_targets['boxes'] = frame_targets['boxes'].squeeze(1)
            ret.append(frame_targets)
        return ret

    def map_global_targets_to_local_targets(self, ret):
        VIS_TrainAPI_clipped_video
        masks = ret['masks'] # N t' h w
        global_obj_appear = masks.flatten(1).any(-1) # N [True, False, True, False, False, False, True]
        ret['masks'] = ret['masks'][global_obj_appear]
        ret['classes'] = ret['classes'][global_obj_appear]
        if 'boxes' in ret:
            ret['boxes'] = ret['boxes'][global_obj_appear] # n t' 4
        return ret
    

class VIS_EvalMapper(VIS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        assert mapper_config['augmentation']['name'] in ['WeakPolyP_EvalAug', 'Visha_EvalAug']
        self.augmentation = VIS_EVAL_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        


