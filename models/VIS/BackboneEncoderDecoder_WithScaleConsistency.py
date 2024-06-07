import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_schedule import build_schedule
from torch import Tensor
from einops import repeat, rearrange, reduce
from functools import partial
from einops.layers.torch import Rearrange
from torch import einsum
import numpy as np
import logging
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann
import torchvision.transforms.functional as Trans_F
import copy
from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from models.backbone.utils import VideoMultiscale_Shape
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY

class BackboneEncoderDecoder_WithScaleConsistency(nn.Module):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.loss_weight = configs['model']['loss_weight']
        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride

        self.fusion_encoder = META_ARCH_REGISTRY.get(configs['model']['fusion']['name'])(configs['model']['fusion'],
                                                                                   multiscale_shapes=self.video_backbone.multiscale_shapes)
          
        same_dim_multiscale_shapes = VideoMultiscale_Shape.set_multiscale_same_dim(shape_by_dim=self.video_backbone.multiscale_shapes,
                                                                                   same_dim=configs['model']['fusion']['d_model'])  
         
        self.decoder = META_ARCH_REGISTRY.get(configs['model']['decoder']['name'])(configs['model']['decoder'],
                                                                                   multiscale_shapes=same_dim_multiscale_shapes)
        if configs['model']['fusion']['name'] == 'Video_Deform2D_DividedTemporal_MultiscaleEncoder_v2':
            self.fusion_encoder.hack_ref(query_norm=self.decoder.temporal_query_norm, mask_mlp=self.decoder.query_mask)
        
        self.test_clip_size = configs['model']['test_clip_size']
    @property
    def device(self):
        return self.pixel_mean.device
    
    def model_preds(self, videos, video_aux_dict,):
        if (not self.training) and (self.test_clip_size is not None):
            nf = videos.shape[2]
            clip_outputs = [] # list[dict]
            for start_idx in range(0, nf, self.test_clip_size):
                multiscales = self.video_backbone(x=videos[:, :, start_idx:(start_idx + self.test_clip_size)]) # b c t h w
                multiscales = self.fusion_encoder(multiscales, video_aux_dict=video_aux_dict) 
                clip_outputs.append(self.decoder(multiscales, video_aux_dict=video_aux_dict)[-1])  # b t nq h w
            return [{
                'pred_masks': torch.cat([haosen['pred_masks'] for haosen in clip_outputs], dim=1), # b t n h w
                'pred_class':  torch.cat([haosen['pred_class'] for haosen in clip_outputs], dim=1),
            }]
        # b 3 t h w -> b 3 t h w
        multiscales = self.video_backbone(x=videos) # b c t h w
        multiscales = self.fusion_encoder(multiscales, video_aux_dict=video_aux_dict) 
        return self.decoder(multiscales, video_aux_dict=video_aux_dict)

    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        videos = batch_dict['video_dict']['videos'] 
        targets = batch_dict['targets']
        batch_size, nf = videos.shape[:2]
        videos = (videos - self.pixel_mean) / self.pixel_std
        size1          = np.random.choice([256, 288, 320, 352, 384, 416, 448])
        vid_1         = F.interpolate(videos.flatten(0, 1), size=size1, mode='bilinear')
        vid_1          = rearrange(vid_1, '(b T) c h w -> b c T h w',b=batch_size, T=nf)
        pred1          = self.model_preds(vid_1, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        pred1_loss = self.decoder.compute_loss(pred1, targets=targets, frame_targets=batch_dict['frame_targets'],
                                               video_aux_dict=batch_dict['video_dict'])
        loss_value_dict = {key: pred1_loss[key] for key in list(self.loss_weight.keys())}
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        videos = videos.permute(0, 2, 1,3,4) # b c t h w
        decoder_output = self.model_preds(videos, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        if isinstance(decoder_output, list):
            decoder_output = decoder_output[-1]
        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear') > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        VIS_Aug_CallbackAPI
        orig_video = videos[0][:, :orig_t, :orig_h, :orig_w].permute(1,0,2,3) # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['base_lr']
        defaults['weight_decay'] = configs['optim']['weight_decay']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )    
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'backbone':None, 'base':None}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if log_lr_group_idx['backbone'] is None:
                        log_lr_group_idx['backbone'] = len(params)

                else:
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                                     
                # pos_embed, norm, embedding的weight decay特殊对待
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    logging.debug(f'setting weight decay of {module_name}.{module_param_name} to zero')
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
   
        return params, log_lr_group_idx

@register_model
def backbone_encoder_decoder_withScaleConsistency(configs, device):
    from .aux_mapper import AUXMapper_v1
    model = BackboneEncoderDecoder_WithScaleConsistency(configs)
    model.to(device)
    params_group, log_lr_group_idx = BackboneEncoderDecoder_WithScaleConsistency.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, f''
    optimizer = get_optimizer(params_group, configs)

    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                model_input_mapper.mapper, 
                                                                partial(model_input_mapper.collate, max_stride=model.max_stride))


    return model, optimizer, scheduler,  train_samplers, train_loaders, log_lr_group_idx, eval_function


