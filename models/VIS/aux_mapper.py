
import torch

from torch.nn import functional as F
from models.registry import register_model
from data_schedule.utils.box_ops import box_xyxy_to_cxcywh
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann
from utils.misc import nested_tensor_from_videos_list_with_stride

class AUXMapper_v1:
    def __init__(self, aux_configs):
        video_auxes = aux_configs['video_auxes']

        video_auxes_names = [config['name'] for config in video_auxes]
        assert len(list(set(video_auxes_names))) == len(video_auxes_names), '每个aux的名字必须不一样'
        self.video_auxes_names = video_auxes_names
        self.video_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in video_auxes]

        self.targets_auxes = None

    def mapper(self, data_dict, mode,):
        if mode == 'train':
            VIS_TrainAPI_clipped_video
            video = data_dict['video_dict']['video']
            for aux, aux_name in zip(self.video_auxes, self.video_auxes_names):
                data_dict['video_dict'][aux_name] = aux.mapper(video)
        
        elif mode == 'evaluate':
            VIS_EvalAPI_clipped_video_request_ann
            video = data_dict['video_dict']['video']
            for aux, aux_name in zip(self.video_auxes, self.video_auxes_names):
                data_dict['video_dict'][aux_name] = aux.mapper(video) 
        else:
            raise ValueError()
           
        return data_dict

    def collate(self, batch_dict, mode, max_stride):
        if mode == 'train':
            VIS_TrainAPI_clipped_video
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride)
            targets = [sample['targets'] for sample in batch_dict]
            frame_has_ann = [clip_tgt['has_ann'] for clip_tgt in targets] # list[t], b
            frame_targets = [sample['frame_targets'] for sample in batch_dict]

            _, pad_T, _, pad_H, pad_W = video_dict['videos'].shape
            targets = self.collate_targets(targets=targets, pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            frame_targets = self.collate_frame_targets(frame_targets=frame_targets, 
                                                                frame_has_ann=frame_has_ann,
                                                                pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            
            ret = {
                'video_dict': video_dict,
                'targets': targets,
                'frame_targets': frame_targets,
                'meta_idxs':  [sample['meta_idx'] for sample in batch_dict],
                'visualize': [sample['visualize'] for sample in batch_dict],               
            }   
                        
        elif mode == 'evaluate':
            VIS_EvalAPI_clipped_video_request_ann
            assert len(batch_dict) == 1
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride) # 不pad
            metas = [sample['meta'] for sample in batch_dict]

            collated_metas = {}
            for key in metas[0].keys():
                collated_metas[key] = [mt[key] for mt in metas]
            ret = {
                'video_dict': video_dict,
                'metas': collated_metas,
                'meta_idxs':  [sample['meta_idx'] for sample in batch_dict],
                'visualize': [sample['visualize'] for sample in batch_dict],  
            }  
        debug_data = False
        if debug_data:
            self.visualize_input_target_for_debug_data(ret) # ./test.png
        return ret

    def collate_video_dict(self, batch_dict, max_stride):
        videos = [sample['video_dict']['video'] for sample in batch_dict]  # list[ti 3 hi wi] -> b T 3 H W
        orig_sizes = [list(vid.shape) for vid in videos] # t 3 h w
        if type(max_stride) == int: # temporal max stride 为1, spatial max stride
            pad_stride = [1, max_stride]
        if (type(max_stride) == list) and (len(max_stride) == 2):
            pad_stride = max_stride
        videos = nested_tensor_from_videos_list_with_stride(videos, max_stride=pad_stride).tensors # b t c h w
        video_dicts = {'videos': videos, 'orig_sizes': orig_sizes}

        for aux_name, aux in zip(self.video_auxes_names, self.video_auxes):
            auxes = [sample['video_dict'][aux_name] for sample in batch_dict] # list[dict] / list[tensor]
            collated_auxes = aux.collate(auxes, batch_videos=videos) # list[dict] / tensor
            if isinstance(auxes[0], dict):
                keys = collated_auxes.keys()
                for key in keys:
                    assert key not in video_dicts
                    video_dicts[key] = collated_auxes[key]
            else:
                video_dicts[aux_name] = collated_auxes

        return video_dicts

    def collate_frame_targets(self, frame_targets, frame_has_ann, pad_H, pad_W, pad_T): # 
        VIS_TrainAPI_clipped_video
        ret = {}
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=0.).bool() for ha in frame_has_ann], dim=0).flatten() # bT
        ret['has_ann'] = has_ann
        masks = [ftarget['masks'] for sample in frame_targets for ftarget in sample] # list[ni h w], bt'
        masks = [F.pad(m.float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2])).bool() for m in masks] # list[ni H W], bt'
        ret['masks'] = masks # list[ni h w], bt'

        classes = [ftarget['classes'] for sample in frame_targets for ftarget in sample] # list[ni], bt'
        ret['classes'] = classes
        
        if 'boxes' in frame_targets[0][0]:
            boxes = [ftarget['boxes'] for sample in frame_targets for ftarget in sample] # list[ni 4], x1y1x2y2, bt'
            boxes = [box_xyxy_to_cxcywh(bx) for bx in boxes]
            boxes = [bx / torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=bx.dtype) for bx in boxes] # 0-1
            ret['boxes'] = boxes # list[ni 4], bt' 
        return ret

    def collate_targets(self, targets, pad_H, pad_W, pad_T):
        VIS_TrainAPI_clipped_video
        has_ann = [sample['has_ann'] for sample in targets] # list[t], bool
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=0.).bool() for ha in has_ann], dim=0) # b T
        masks = [sample['masks'] for sample in targets] 
        masks = [F.pad(m.float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2]), value=0.).bool() \
                 for m in masks] # list[ni T' H W]
        classes = [sample['classes'] for sample in targets]
        ret = {
            'masks': masks, # list[ni T' h w]
            'has_ann': has_ann, # b T
            'classes': classes, # list[ni], b
        } 
        if 'boxes' in targets[0]:
            boxes = [sample['boxes'] for sample in targets] # list[ni T' 4], x1y1x2y2
            boxes = [box_xyxy_to_cxcywh(bx) for bx in boxes]
            boxes = [bx / torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=torch.float) for bx in boxes] # 0-1
            ret.update({'boxes': boxes,})
        return ret

    def visualize_input_target_for_debug_data(self, ret):
        videos = ret['video_dict']['videos'] # b T 3 H W
        pass
