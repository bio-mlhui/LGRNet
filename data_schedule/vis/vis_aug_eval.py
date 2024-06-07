# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import torch
import torchvision.transforms.functional as F
from data_schedule.vis.apis import VIS_Aug_CallbackAPI
from .vis_aug_utils import  get_tgt_size

from .vis_aug_utils import VIS_EVAL_AUG_REGISTRY

class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, ret):
        video = ret['video']
        orig_size = video[0].size # w h
        tgt_size = get_tgt_size(video[0].size, random.choice(self.sizes), self.max_size) # h w

        resized_video = [F.resize(frame, tgt_size) for frame in video]
        ratio_width, ratio_height = tuple(float(s) / float(s_orig) for s, s_orig in zip(tgt_size[::-1], orig_size))
        ret['video'] = resized_video

        if 'callback_fns' in ret:
            VIS_Aug_CallbackAPI
            ret['callback_fns'].append(RandomResize(sizes=[orig_size], max_size=None))

        if "pred_masks" in ret:
            assert (len(self.sizes) == 1) and (self.max_size == None)
            VIS_Aug_CallbackAPI
            pred_masks = ret['pred_masks'] # list[nt h w], t
            pred_masks = [torch.nn.functional.interpolate(mk.unsqueeze(0).float(), tgt_size, mode='nearest')[0].bool()
                          for mk in pred_masks]
            ret['pred_masks'] = pred_masks # list[nt h w], t

        if "pred_boxes" in ret:
            VIS_Aug_CallbackAPI
            pred_boxes = ret["pred_boxes"] # list[nt 4], t
            scaled_boxes = [bx * (torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])[None, :])
                            for bx in pred_boxes]
            ret["pred_boxes"] = scaled_boxes

        return ret
    
class VideoToPIL:
    def __call__(self, ret):
        video = ret['video'] # t 3 h w ->
        assert video.dtype == torch.float and (video.max() <= 1) and (video.min() >=0)  
        pil_video = [F.to_pil_image(frame, mode='RGB') for frame in video] # 3 h w, float, 0-1
        ret['video'] = pil_video
        assert 'callback_fns' not in ret
        return ret


class VideoToTensor:
    def __call__(self, ret):
        video = ret['video']
        tensor_video = torch.stack([F.to_tensor(frame) for frame in video], dim=0) # t 3 h w, float, 0-1
        ret['video'] = tensor_video

        if 'callback_fns' in ret:
            VIS_Aug_CallbackAPI
            ret['callback_fns'].append(VideoToPIL())

        return ret
        

@VIS_EVAL_AUG_REGISTRY.register()
class WeakPolyP_EvalAug:
    def __init__(self, configs) -> None:
        self.resize = RandomResize(
            sizes=[[352, 352]],
        )
        self.tensor_video = VideoToTensor()

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret
